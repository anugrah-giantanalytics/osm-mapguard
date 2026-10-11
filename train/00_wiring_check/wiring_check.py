"""[WIRING CHECK · no results] Prove every training stage runs on this machine and image.

Run order 0. Run it first on every new machine, container image or TensorFlow version.

    python train/00_wiring_check/wiring_check.py
    python train/00_wiring_check/wiring_check.py --device cpu

**This is not data and produces no results.** It pushes random tensors of the right shapes
through the real code paths - the same `build_model`, `train_stage2`, normaliser,
attribution, Stage 1 encoder and weight transfer that the real stages use - for one tiny
epoch each, inside a temporary directory that is deleted afterwards. Nothing is written to
`experiments/`, `results/` or `experiments/splits/`. Its only output is PASS or FAIL.

What a FAIL here usually means:
  - a Keras 2 / Keras 3 incompatibility in the model graph (see stage2.py header)
  - masking that leaks - padded edits changing the prediction
  - weights that do not round-trip through save and rebuild
  - Stage 1 encoder weights that do not fit Stage 2's `edit_encoder`

The random tensors are shaped to exercise the awkward cases on purpose: changesets with no
edits at all, users with no history, heavy-tailed geometry, both classes.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN / "04_stage1_pretrain"), str(TRAIN), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

from common.preprocess import (  # noqa: E402
    Normaliser,
    Prepared,
    select_branches,
    signed_log1p,
)
from common.run import Run, configure_device, keras_version, seed_everything  # noqa: E402
from mapguard.training.dataset import (  # noqa: E402
    EDIT_FEATURES,
    EDIT_TEXT_LEN,
    TABULAR_FEATURES,
    TOKEN_BUCKETS,
    TOKENS_PER_EDIT,
    VOCAB_SIZE,
)

N, K, S, T = 240, 16, 12, 256


def random_prepared(seed: int = 7) -> Prepared:
    """Random tensors in the feature store's shapes. Wiring only - never a result."""
    rng = np.random.default_rng(seed)
    F = len(EDIT_FEATURES)
    n_edits = rng.integers(0, K + 1, N)
    n_edits[:6] = 0                                   # empty changesets must not produce NaN
    edit_mask = (np.arange(K)[None, :] < n_edits[:, None]).astype("float32")
    edit = rng.lognormal(1.0, 3.0, size=(N, K, F)) * rng.choice([-1, 1], size=(N, K, F))
    edit = (edit * edit_mask[:, :, None]).astype("float32")

    n_hist = rng.integers(0, S + 1, N)
    n_hist[:10] = 0                                   # first-ever edits: no history
    user_mask = (np.arange(S)[None, :] < n_hist[:, None]).astype("float32")
    user = (rng.lognormal(0.5, 2.0, size=(N, S, 4)) * user_mask[:, :, None]).astype("float32")

    return Prepared(
        arrays={
            "edit_features": edit, "edit_mask": edit_mask,
            "edit_tokens": (rng.integers(1, TOKEN_BUCKETS, size=(N, K, TOKENS_PER_EDIT))
                            * edit_mask[:, :, None]).astype("int32"),
            "edit_text": (rng.integers(0, VOCAB_SIZE, size=(N, K, EDIT_TEXT_LEN))
                          * edit_mask[:, :, None]).astype("int32"),
            "text": rng.integers(0, VOCAB_SIZE, size=(N, T)).astype("int32"),
            "user_sequence": user, "user_mask": user_mask,
            "tabular": rng.lognormal(0.0, 2.0, size=(N, len(TABULAR_FEATURES))).astype("float32"),
        },
        y=np.tile([0.0, 1.0], N // 2).astype("float32"),
        cids=np.arange(1_000_000, 1_000_000 + N),
        uids=np.array([int(u) for u in rng.integers(1, 60, N)], dtype=object),
        created_at=np.array([f"2025-{1 + i % 12:02d}-01T00:00:00Z" for i in range(N)]),
        edit_columns=tuple(EDIT_FEATURES),
        tabular_columns=tuple(TABULAR_FEATURES),
    )


def masks_for(n: int) -> dict:
    idx = np.arange(n)
    return {"train": idx % 5 < 3, "val": idx % 5 == 3, "test": idx % 5 == 4}


def base_cfg(branches) -> dict:
    return {
        "stage": "wiring_check", "task": "wiring", "split": "wiring", "seed": 1,
        "model": {"branches": list(branches), "max_edits": K, "user_steps": S,
                  "embed": 32, "heads": 2, "dropout": 0.1, "raster": None},
        "train": {"epochs": 1, "batch_size": 32, "lr": 1e-3, "loss": "focal",
                  "patience": 1, "verbose": 0, "author_weighting": "sqrt",
                  "weight_decay": 1e-4},
        "evaluate": {"attribution": True, "top_k": 3},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="auto", choices=("auto", "cpu"))
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    ap.add_argument("--mixed", action="store_true",
                    help="also train the full model under mixed precision (run_pipeline --fast)")
    args = ap.parse_args()

    devices = configure_device(args.device)
    import tensorflow as tf
    from tensorflow import keras

    from mapguard.models.stage2 import pack
    from common.fit import ENCODER_FILE, build, train_stage2
    from corruptions import CORRUPTIONS, make_contrast_set

    tmp = Path(tempfile.mkdtemp(prefix="mapguard_wiring_"))
    checks: list[dict] = []

    def check(name, fn):
        t0 = time.monotonic()
        try:
            detail = fn()
            checks.append({"check": name, "ok": True, "seconds": round(time.monotonic() - t0, 1),
                           "detail": detail})
            print(f"PASS  {name}  ({checks[-1]['seconds']}s)")
        except BaseException as exc:          # SystemExit included - the stages use it to refuse
            checks.append({"check": name, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "trace": traceback.format_exc()[-1500:]})
            print(f"FAIL  {name}\n      {type(exc).__name__}: {exc}")

    raw = random_prepared()
    masks = masks_for(len(raw.y))
    seed_everything(1)

    # 1. every ablation branch set builds, trains, masks, attributes -------------------
    branch_sets = [["B5"], ["B1", "B5"], ["B2", "B5"], ["B3", "B4", "B5"], ["B5", "B6"],
                   ["B1", "B2", "B3", "B4", "B5", "B6"]]
    full_run_dir = {}
    for bs in branch_sets:
        def one(bs=bs):
            tf.keras.backend.clear_session()
            cfg = base_cfg(bs)
            prep = select_branches(raw, bs)
            run = Run(cfg, tag="_".join(bs), root=tmp)
            res = train_stage2(cfg, prep, masks, run)
            for when in ("before_training", "after_training"):
                if not res["masking_checks"][when]["masking_holds"]:
                    raise AssertionError(f"masking leaks {when}")
            if not np.isfinite(res["headline"]["test_pr_auc"][0]):
                raise AssertionError("non-finite test PR-AUC")
            full_run_dir["dir"] = run.dir
            return {"params": res["training"]["params"],
                    "masking_delta_after": res["masking_checks"]["after_training"]["max_delta"],
                    "null_edits_inserted": res["notes"]}
        check(f"stage2 train {'+'.join(bs)}", one)

    def meanpool():
        tf.keras.backend.clear_session()
        cfg = base_cfg(branch_sets[-1])
        cfg["model"]["pooling"] = "mean"
        run = Run(cfg, tag="meanpool", root=tmp)
        res = train_stage2(cfg, select_branches(raw, branch_sets[-1]), masks, run)
        if not res["masking_checks"]["after_training"]["masking_holds"]:
            raise AssertionError("masking leaks with mean pooling")
        return {"pooling": res["training"]["pooling"]}
    check("stage2 full model with round-1 mean pooling", meanpool)

    def v3():
        """v3 options together: pretrained text embeddings as an input, two transformer blocks
        over edits, and fold scoring for stacking (extra_rows)."""
        from dataclasses import replace as dc_replace

        tf.keras.backend.clear_session()
        cfg = base_cfg(branch_sets[-1])
        cfg["model"]["edit_layers"] = 2
        cfg["train"]["mixed_precision"] = "auto"
        prep = select_branches(raw, branch_sets[-1])
        emb = np.random.default_rng(3).normal(size=(len(prep.y), 768)).astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        prep = dc_replace(prep, arrays={**prep.arrays, "text_emb": emb})
        held = np.zeros(len(prep.y), bool)
        held[np.flatnonzero(masks["train"])[:20]] = True
        fold_train = masks["train"] & ~held
        run = Run(cfg, tag="v3", root=tmp)
        try:
            res = train_stage2(cfg, prep, masks, run, train_rows=fold_train, extra_rows=held)
        finally:
            tf.keras.mixed_precision.set_global_policy("float32")
        for when in ("before_training", "after_training"):
            if not res["masking_checks"][when]["masking_holds"]:
                raise AssertionError(f"masking leaks with v3 options {when}")
        z = np.load(run.path("predictions.npz"))
        if len(z["extra_p"]) != 20 or not np.isfinite(z["extra_p"]).all():
            raise AssertionError("fold scoring (extra_rows) did not return 20 finite scores")
        kw = json.loads(run.path("model_config.json").read_text())["build_kwargs"]
        if kw.get("text_emb_dim") != 768 or kw.get("edit_layers") != 2:
            raise AssertionError(f"v3 options not recorded in model_config: {kw}")
        tf.keras.backend.clear_session()
        m = build(kw)
        m.load_weights(str(run.path("model.weights.h5")))
        if "text_emb" not in [n for n in (t.name.split(":")[0] for t in m.inputs)]:
            raise AssertionError("rebuilt model has no text_emb input")
        return {"params": res["training"]["params"], "precision": res["training"]["precision"],
                "extra_scored": int(len(z["extra_p"]))}
    check("v3: text embeddings + 2 transformer blocks + fold scoring", v3)

    # 2. weights round-trip through model_config + rebuild -----------------------------
    def roundtrip():
        tf.keras.backend.clear_session()
        d = full_run_dir["dir"]
        mcfg = json.loads((d / "model_config.json").read_text())
        prep = Normaliser.load(d / "normaliser.json").apply(select_branches(raw, branch_sets[-1]))
        m1 = build(mcfg["build_kwargs"]); m1.load_weights(str(d / "model.weights.h5"))
        m2 = build(mcfg["build_kwargs"]); m2.load_weights(str(d / "model.weights.h5"))
        p1 = m1.predict(pack(m1, prep.arrays), verbose=0)[0]
        p2 = m2.predict(pack(m2, prep.arrays), verbose=0)[0]
        if not np.allclose(p1, p2, atol=1e-6):
            raise AssertionError("two rebuilds from the same weights disagree")
        return {"max_abs_diff": float(np.max(np.abs(p1 - p2)))}
    check("weights save -> rebuild -> load", roundtrip)

    # 3. Stage 1 pretrain on raw edits, then transfer into Stage 2 ---------------------
    enc_dir = tmp / "stage1_encoder"

    def stage1():
        tf.keras.backend.clear_session()
        enc_dir.mkdir(parents=True, exist_ok=True)
        cols = list(raw.edit_columns)
        rows = masks["train"]
        real = raw.arrays["edit_features"][rows][raw.arrays["edit_mask"][rows] > 0]
        X, y, kinds = make_contrast_set(real, cols, seed=1, kinds=CORRUPTIONS)
        norm = Normaliser().fit(raw, rows, groups=("edit",))
        norm.save(enc_dir / "normaliser.json")
        Xn = norm._z(signed_log1p(X), "edit", cols).astype("float32")
        inp = keras.Input(shape=(len(cols),))
        z = keras.layers.Dense(32, activation="relu", name="encoder")(inp)
        out = keras.layers.Dense(1, activation="sigmoid")(z)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(Xn, y, epochs=1, batch_size=256, verbose=0)
        kernel, bias = model.get_layer("encoder").get_weights()
        np.savez(enc_dir / ENCODER_FILE, kernel=kernel, bias=bias, edit_columns=np.array(cols),
                 embed=np.array(32))
        (enc_dir / "config.json").write_text(json.dumps({"task": "wiring", "split": "wiring", "seed": 1}))
        return {"contrast_rows": int(len(X)), "corruption_kinds_used": sorted(set(int(k) for k in kinds if k >= 0))}
    check("stage1 corruptions + encoder pretrain", stage1)

    def transfer():
        tf.keras.backend.clear_session()
        cfg = base_cfg(branch_sets[-1])
        cfg["init"] = {"pretrained_encoder": str(enc_dir), "freeze_encoder": False}
        prep = select_branches(raw, branch_sets[-1])
        run = Run(cfg, tag="pretrained", root=tmp)
        res = train_stage2(cfg, prep, masks, run)
        return {"encoder": res["pretrained_encoder"]}
    check("stage1 encoder -> stage2 edit_encoder transfer", transfer)

    if args.mixed or tf.config.list_physical_devices("GPU"):
        def mixed():
            tf.keras.backend.clear_session()
            cfg = base_cfg(branch_sets[-1])
            cfg["train"]["mixed_precision"] = "on"
            run = Run(cfg, tag="mixed", root=tmp)
            res = train_stage2(cfg, select_branches(raw, branch_sets[-1]), masks, run)
            tf.keras.mixed_precision.set_global_policy("float32")
            for when in ("before_training", "after_training"):
                if not res["masking_checks"][when]["masking_holds"]:
                    raise AssertionError(f"masking leaks under mixed precision {when}")
            if not np.isfinite(res["headline"]["test_pr_auc"][0]):
                raise AssertionError("non-finite PR-AUC under mixed precision")
            return {"precision": res["training"]["precision"],
                    "masking_delta_after": res["masking_checks"]["after_training"]["max_delta"]}
        check("stage2 full model under mixed precision", mixed)

    def wrong_split_refused():
        cfg = base_cfg(branch_sets[-1])
        cfg["task"] = "a_different_task"
        cfg["init"] = {"pretrained_encoder": str(enc_dir)}
        run = Run(cfg, tag="should_refuse", root=tmp)
        try:
            train_stage2(cfg, select_branches(raw, branch_sets[-1]), masks, run)
        except SystemExit as exc:
            return {"refused_with": str(exc)[:120]}
        raise AssertionError("an encoder from a different split was accepted")
    check("encoder from a different split is refused", wrong_split_refused)

    ok = all(c["ok"] for c in checks)
    summary = {
        "verdict": "PASS" if ok else "FAIL",
        "tensorflow": tf.__version__, "keras": keras_version(),
        "devices": devices, "checks": checks,
        "note": "random tensors, wiring only - not a result, nothing written outside a temp dir",
    }
    print("\n" + json.dumps({k: summary[k] for k in ("verdict", "tensorflow", "keras", "devices")}, indent=2))
    if args.keep:
        (tmp / "wiring_check.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"kept: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
