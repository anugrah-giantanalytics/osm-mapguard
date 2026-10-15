"""[STAGE 2 · evaluation] Re-evaluate a trained Stage 2 run, optionally on another task.

Run order 2 of 6 (companion to stage2_train.py). Plan stage: Stage 2.

    # re-score a run on its own test fold
    python train/02_stage2_supervised/stage2_evaluate.py --run experiments/runs/stage2_supervised/<dir>

    # the T1/T2 gap: a model trained on T2 (detection) scored on T1 (triage)
    python train/02_stage2_supervised/stage2_evaluate.py --run <T2 run dir> --task T1

The gap between T1 and T2 is a result, not a nuisance: it measures how much of a score is
carried by OSMCha's flagging heuristic - both T1 classes come from the flagged tail -
rather than by the model. See docs/data/DATA_FOUNDATION.md section 1.

The **threshold is not re-chosen** on the target task. It comes from the source run's
validation fold. Re-picking it on the target test set would flatter the transfer.

Everything needed is read from the run directory: config, model_config, normaliser,
weights. Nothing is retrained.
"""

import argparse
import json
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

from common import data_contract, preprocess, scoring, splits_io  # noqa: E402
from common.attribution import branch_attribution  # noqa: E402
from common.fit import build  # noqa: E402
from common.preprocess import Normaliser  # noqa: E402
from common.run import Run, configure_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a stage2_supervised run directory")
    ap.add_argument("--task", default=None, help="evaluate on this task instead (T1, T2, ovid)")
    ap.add_argument("--split", default=None, help="which frozen split's test fold to use")
    ap.add_argument("--no-attribution", action="store_true")
    args = ap.parse_args()

    src = Path(args.run)
    src_cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    mcfg = json.loads((src / "model_config.json").read_text(encoding="utf-8"))
    src_results = json.loads((src / "results.json").read_text(encoding="utf-8"))
    threshold = src_results["evaluation"]["threshold"]["value"]

    task = args.task or src_cfg["task"]
    split = args.split or src_cfg["split"]
    seed = int(src_cfg.get("split_seed", src_cfg.get("seed", 0)))

    cfg = {**src_cfg, "stage": "stage2_evaluation", "task": task, "split": split,
           "source_run": str(src), "tag": f"from_{src_cfg['task']}"}
    devices = configure_device(src_cfg.get("device", "auto"))

    from mapguard.models.stage2 import pack, verify_masking

    store = data_contract.load(src_cfg.get("data", {}).get("feature_store"), task=task)
    prep = preprocess.prepare(store, src_cfg)
    if list(prep.edit_columns) != mcfg["edit_columns"] or list(prep.tabular_columns) != mcfg["tabular_columns"]:
        raise SystemExit("feature columns differ from the ones this model was trained on; "
                         "it cannot be evaluated on this feature store")

    masks, split_path, split_summary = splits_io.get_split(
        prep, task=task, split=split, seed=seed,
        cutoff=src_cfg.get("temporal", {}).get("cutoff"))
    test = masks["test"].copy()

    # Cross-task leak guard. T1 and T2 share the same positives (corpus M, harmful), so a
    # T1 test fold can contain changesets the T2 model trained on. Remove every row whose
    # cid was in the source run's train or validation fold, and - for a user-disjoint
    # source - every row from a user the source trained on. Without this the transfer
    # number is inflated by memorised rows.
    removed = {"by_cid": 0, "by_uid": 0}
    src_prov = json.loads((src / "provenance.json").read_text(encoding="utf-8"))
    if src_prov.get("split_file") and Path(src_prov["split_file"]).exists():
        sdoc = json.loads(Path(src_prov["split_file"]).read_text(encoding="utf-8"))
        seen = set(sdoc["train"]) | set(sdoc["val"])
        by_cid = np.array([int(c) in seen for c in prep.cids])
        removed["by_cid"] = int((test & by_cid).sum())
        test &= ~by_cid
        if src_cfg["split"] != "R1a":
            src_uids = {u for u, c in zip(prep.uids, prep.cids) if int(c) in seen and u is not None}
            by_uid = np.array([u in src_uids for u in prep.uids])
            removed["by_uid"] = int((test & by_uid).sum())
            test &= ~by_uid
    elif task != src_cfg["task"]:
        raise SystemExit("cannot find the source run's split file, so cross-task evaluation "
                         "cannot exclude rows the model trained on - refusing")
    if test.sum() == 0 or len(np.unique(prep.y[test])) < 2:
        raise SystemExit(f"after removing rows the source model saw, the test fold is unusable: {removed}")

    prep = Normaliser.load(src / "normaliser.json").apply(prep)
    model = build(mcfg["build_kwargs"])
    model.load_weights(str(src / "model.weights.h5"))

    a, y = prep.arrays, prep.y
    check = verify_masking(model, {k: v[test] for k, v in a.items()})
    if not check["masking_holds"]:
        raise SystemExit(f"masking check failed: {check}")

    p, w = model.predict(pack(model, a, test), batch_size=256, verbose=0)
    p = p.ravel()

    from mapguard.evaluation.metrics import average_precision, bootstrap_ci, roc_auc, summary

    pr = bootstrap_ci(y[test], p, metric=average_precision, seed=seed)
    roc = bootstrap_ci(y[test], p, metric=roc_auc, seed=seed)
    evaluation = {
        "threshold": {"value": threshold, "chosen_on": f"validation of source run ({src_cfg['task']})"},
        "test": summary(y[test], p, threshold=threshold),
        "test_intervals_95": {"pr_auc": [round(x, 4) for x in pr], "roc_auc": [round(x, 4) for x in roc]},
        "test_calibration": scoring.calibration(y[test], p),
    }

    attribution = None
    if not args.no_attribution:
        attribution, _, _ = branch_attribution(
            model, {k: v[test] for k, v in a.items()}, branches=mcfg["build_kwargs"]["branches"],
            edit_columns=prep.edit_columns, tabular_columns=prep.tabular_columns,
            y=y[test], seed=seed)

    run = Run(cfg)
    run.provenance(devices=devices, data_files=store.files, split_file=split_path,
                   extra={"source_run": str(src), "split_summary": split_summary})
    np.savez_compressed(run.path("predictions.npz"), cids=prep.cids[test], y=y[test], p=p)

    source_head = src_results["headline"]
    results = {
        "source_run": str(src),
        "trained_on": {"task": src_cfg["task"], "split": src_cfg["split"]},
        "evaluated_on": {"task": task, "split": split},
        "headline": scoring.headline(evaluation),
        "source_headline": source_head,
        "pr_auc_change_vs_source": round(evaluation["test_intervals_95"]["pr_auc"][0]
                                         - source_head["test_pr_auc"][0], 4),
        "evaluation": evaluation,
        "attribution": attribution,
        "masking_check": check,
        "rows_removed_seen_by_source_model": removed,
        "n_test_after_removal": int(test.sum()),
    }
    run.finish(results)
    print(json.dumps({k: results[k] for k in ("trained_on", "evaluated_on", "headline",
                                              "pr_auc_change_vs_source")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
