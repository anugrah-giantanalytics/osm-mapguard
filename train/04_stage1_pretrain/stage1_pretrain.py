"""[STAGE 1 · self-supervised] Pretrain the edit encoder by corruption contrast.

Run order 4 of 6. Plan stage: **Stage 1** - conditional, week 4 (PROJECT_PLAN §13). Cluster.

    python train/04_stage1_pretrain/stage1_pretrain.py
    python train/04_stage1_pretrain/stage1_pretrain.py --dry-run

The encoder learns "normal edit vs damaged edit" from real edits and synthetic damage
(see corruptions.py). Its first layer is exactly the `edit_encoder` layer of the Stage 2
model - same input columns, same width, same normalisation - so the weights transfer
by name. Anything that differs would make "pretrained" a label on a random initialisation.

**Evaluation data never reaches pretraining.** Edits come only from changesets in the
training fold of the frozen split, minus any changeset whose author also appears in the
validation or test folds. The overlap counts are written to `disjointness.json`, and the
run aborts if either is non-zero. Pretraining on the test set would make the label-
efficiency curve (R4) - the most important figure in the plan - meaningless.

Pretraining validation is split **by changeset**, not by edit: edits from one changeset are
near-duplicates, and splitting them across folds would report a memorised number.

Writes experiments/runs/stage1_pretrain/<dir>/ with edit_encoder.npz and normaliser.json -
the two files stage 02/03/05 need via `init.pretrained_encoder`.
"""

import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(TRAIN), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

from common import stage  # noqa: E402
from common.fit import ENCODER_FILE  # noqa: E402
from common.preprocess import Normaliser, signed_log1p  # noqa: E402
from corruptions import CORRUPTIONS, make_contrast_set  # noqa: E402


def pretraining_rows(prep, masks) -> tuple[np.ndarray, dict]:
    """Training-fold changesets whose authors never appear in validation or test."""
    held_out = masks["val"] | masks["test"]
    held_users = {u for u in prep.uids[held_out] if u is not None}
    rows = masks["train"] & ~np.array([u in held_users for u in prep.uids])

    held_cids = set(prep.cids[held_out].tolist())
    proof = {
        "pretraining_changesets": int(rows.sum()),
        "excluded_train_changesets_sharing_an_author_with_eval": int((masks["train"] & ~rows).sum()),
        "overlap_changeset_ids": int(sum(1 for c in prep.cids[rows] if c in held_cids)),
        "overlap_user_ids": int(sum(1 for u in prep.uids[rows] if u is not None and u in held_users)),
    }
    if proof["overlap_changeset_ids"] or proof["overlap_user_ids"]:
        raise SystemExit(f"pretraining overlaps evaluation data - refusing: {proof}")
    return rows, proof


def collect_edits(prep, rows, max_edits: int, seed: int):
    """Raw real edit vectors plus the changeset each came from."""
    ef, em = prep.arrays["edit_features"][rows], prep.arrays["edit_mask"][rows]
    owners = np.repeat(prep.cids[rows], em.shape[1]).reshape(em.shape)
    real = em > 0
    X, own = ef[real], owners[real]
    if len(X) > max_edits:
        pick = np.random.default_rng(seed).choice(len(X), max_edits, replace=False)
        X, own = X[pick], own[pick]
    return X, own


def main() -> int:
    args = stage.parse_args(__doc__.splitlines()[0], HERE / "config.json")
    cfg, run, store, prep, masks = stage.setup(args)

    import tensorflow as tf
    from tensorflow import keras

    p_cfg, m_cfg = cfg.get("pretrain", {}), cfg.get("model", {})
    seed, embed = int(cfg["seed"]), int(m_cfg.get("embed", 64))

    rows, proof = pretraining_rows(prep, masks)
    run.write_json("disjointness.json", proof)
    print("disjointness:", proof)

    # Normalisation is fitted on the same rows and columns Stage 2 will use, and saved:
    # Stage 2 loads these edit statistics instead of refitting when it uses this encoder.
    norm = Normaliser().fit(prep, rows, groups=("edit",))
    norm.save(run.path("normaliser.json"))
    cols = list(prep.edit_columns)

    X_raw, owners = collect_edits(prep, rows, int(p_cfg.get("max_edits", 400_000)), seed)
    uniq = np.unique(owners)
    rng = np.random.default_rng(seed)
    val_owners = set(rng.choice(uniq, max(1, int(len(uniq) * float(p_cfg.get("val_fraction", 0.1)))),
                                replace=False).tolist())
    is_val = np.array([o in val_owners for o in owners])

    kinds = tuple(p_cfg.get("corruptions", CORRUPTIONS))
    Xtr, ytr, ktr = make_contrast_set(X_raw[~is_val], cols, seed=seed, kinds=kinds)
    Xva, yva, kva = make_contrast_set(X_raw[is_val], cols, seed=seed + 1, kinds=kinds)

    def transform(X):
        return norm._z(signed_log1p(X), "edit", cols).astype("float32")

    Xtr, Xva = transform(Xtr), transform(Xva)
    print(f"contrast set: train {len(Xtr):,}  val {len(Xva):,}  features {len(cols)}")

    inp = keras.Input(shape=(len(cols),), name="edit_features")
    z = keras.layers.Dense(embed, activation="relu", name="encoder")(inp)
    h = keras.layers.Dropout(float(p_cfg.get("dropout", 0.1)))(z)
    h = keras.layers.Dense(64, activation="relu", name="head_hidden")(h)
    out = keras.layers.Dense(1, activation="sigmoid", name="corrupted")(h)
    model = keras.Model(inp, out, name="stage1_corruption_contrast")
    model.compile(optimizer=keras.optimizers.Adam(float(p_cfg.get("lr", 1e-3))),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")])

    hist = model.fit(
        Xtr, ytr, validation_data=(Xva, yva),
        epochs=int(p_cfg.get("epochs", 30)), batch_size=int(p_cfg.get("batch_size", 1024)),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                          patience=int(p_cfg.get("patience", 4)),
                                          restore_best_weights=True),
            keras.callbacks.CSVLogger(str(run.path("training_log.csv"))),
        ],
        verbose=int(p_cfg.get("verbose", 2)),
    )

    p_va = model.predict(Xva, batch_size=4096, verbose=0).ravel()
    from mapguard.evaluation.metrics import roc_auc

    per_kind = {}
    for j, name in enumerate(kinds):
        m = kva == j
        if m.any():
            per_kind[name] = {"n": int(m.sum()), "detected_at_0.5": round(float((p_va[m] >= 0.5).mean()), 4)}
    real = kva == -1
    per_kind["real_edits_false_alarm_at_0.5"] = round(float((p_va[real] >= 0.5).mean()), 4)

    kernel, bias = model.get_layer("encoder").get_weights()
    np.savez(run.path(ENCODER_FILE), kernel=kernel, bias=bias,
             edit_columns=np.array(cols), embed=np.array(embed))
    model.save_weights(str(run.path("contrast_model.weights.h5")))
    run.write_json("model_config.json", {"edit_columns": cols, "embed": embed,
                                         "dropout": float(p_cfg.get("dropout", 0.1)),
                                         "corruptions": list(kinds)})

    val_auc = round(float(roc_auc(yva, p_va)), 4)
    run.finish({
        "headline": {"pretext_val_auc": val_auc,
                     "weakest_corruption": min(((k, v["detected_at_0.5"]) for k, v in per_kind.items()
                                                if isinstance(v, dict)), key=lambda kv: kv[1])[0]},
        "pretext": {"val_auc": val_auc, "per_corruption": per_kind,
                    "epochs_run": len(hist.history["loss"]), "n_train": int(len(Xtr)),
                    "n_val": int(len(Xva)), "corruptions": list(kinds)},
        "disjointness": proof,
        "known_gaps": ["no text corruption - tag text is changeset-level in this feature store",
                       "no mass-deletion corruption - deletion volume is changeset-level"],
        "encoder": {"file": ENCODER_FILE, "embed": embed, "edit_columns": cols},
        "tensorflow": tf.__version__, "keras": keras.__version__,
        "next": "run stage1_zero_shot.py on this directory before using it to fine-tune",
    })
    print(f"\npretext val AUC {val_auc}. A near-perfect pretext score says nothing about "
          "vandalism - run stage1_zero_shot.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
