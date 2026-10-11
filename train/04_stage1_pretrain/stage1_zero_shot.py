"""[STAGE 1 · transfer check] Score real labelled changesets with the frozen pretext model.

Run order 4 of 6 (companion to stage1_pretrain.py). Plan stage: Stage 1. Run this
**before** using an encoder for fine-tuning.

    python train/04_stage1_pretrain/stage1_zero_shot.py --run experiments/runs/stage1_pretrain/<dir>

A model can separate real edits from synthetic damage perfectly and still know nothing
about vandalism - if the corruptions are the wrong distribution, it learned the corruptions.
This checks that directly: no fine-tuning, just "how damaged does the pretext model think
each edit is", aggregated to the changeset, against the real labels on the test fold.

    AUC ~ 0.5   the corruptions do not resemble real damage - redesign before fine-tuning
    AUC > 0.6   the pretext task carries some real signal

**Report the number either way.** It is a finding about the pretext task, not a diagnostic
to hide when it disappoints.
"""

import argparse
import json
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

from common import data_contract, preprocess, splits_io  # noqa: E402
from common.preprocess import Normaliser, signed_log1p  # noqa: E402
from common.run import Run, configure_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a stage1_pretrain run directory")
    args = ap.parse_args()

    src = Path(args.run)
    src_cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    mcfg = json.loads((src / "model_config.json").read_text(encoding="utf-8"))
    devices = configure_device(src_cfg.get("device", "auto"))

    from tensorflow import keras

    from mapguard.evaluation.metrics import average_precision, bootstrap_ci, roc_auc

    task, split, seed = src_cfg["task"], src_cfg["split"], int(src_cfg.get("split_seed", src_cfg["seed"]))
    store = data_contract.load(src_cfg.get("data", {}).get("feature_store"), task=task)
    prep = preprocess.prepare(store, src_cfg)
    if list(prep.edit_columns) != mcfg["edit_columns"]:
        raise SystemExit("edit columns differ from the pretrained model's")
    masks, split_path, _ = splits_io.get_split(prep, task=task, split=split, seed=seed,
                                               cutoff=src_cfg.get("temporal", {}).get("cutoff"))
    test = masks["test"]

    norm = Normaliser.load(src / "normaliser.json", groups=("edit",))
    cols = mcfg["edit_columns"]
    inp = keras.Input(shape=(len(cols),), name="edit_features")
    z = keras.layers.Dense(mcfg["embed"], activation="relu", name="encoder")(inp)
    h = keras.layers.Dropout(mcfg["dropout"])(z)
    h = keras.layers.Dense(64, activation="relu", name="head_hidden")(h)
    out = keras.layers.Dense(1, activation="sigmoid", name="corrupted")(h)
    model = keras.Model(inp, out)
    model.load_weights(str(src / "contrast_model.weights.h5"))

    ef = prep.arrays["edit_features"][test]
    em = prep.arrays["edit_mask"][test]
    y = prep.y[test]
    n, k, f = ef.shape
    flat = norm._z(signed_log1p(ef.reshape(-1, f)), "edit", cols).astype("float32")
    scores = model.predict(flat, batch_size=8192, verbose=0).reshape(n, k)
    real = em > 0
    has_edits = real.any(axis=1)

    max_score = np.where(real, scores, -np.inf).max(axis=1)
    mean_score = np.where(real, scores, 0.0).sum(axis=1) / np.maximum(real.sum(axis=1), 1)
    ys, mx, mn = y[has_edits], max_score[has_edits], mean_score[has_edits]

    def block(s):
        pr = bootstrap_ci(ys, s, metric=average_precision, seed=seed)
        roc = bootstrap_ci(ys, s, metric=roc_auc, seed=seed)
        return {"roc_auc_ci95": [round(v, 4) for v in roc], "pr_auc_ci95": [round(v, 4) for v in pr]}

    results = {
        "source_run": str(src),
        "n_test_changesets": int(len(y)),
        "n_scored_with_edits": int(has_edits.sum()),
        "aggregate_max_edit_score": block(mx),
        "aggregate_mean_edit_score": block(mn),
    }
    best_auc = max(results["aggregate_max_edit_score"]["roc_auc_ci95"][0],
                   results["aggregate_mean_edit_score"]["roc_auc_ci95"][0])
    results["verdict"] = ("pretext carries real signal" if best_auc >= 0.6 else
                          "near chance - the corruptions do not resemble real damage; "
                          "redesign them before fine-tuning")
    results["headline"] = {"zero_shot_roc_auc": round(best_auc, 4), "verdict": results["verdict"]}

    cfg = {**src_cfg, "stage": "stage1_zero_shot", "source_run": str(src)}
    run = Run(cfg)
    run.provenance(devices=devices, data_files=store.files, split_file=split_path,
                   extra={"source_run": str(src)})
    run.finish(results)
    print(json.dumps(results["headline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
