"""[STAGE 2 · R3 temporal] Train on changesets before a cutoff, test on changesets after it.

Run order 6 of 6. Plan stage: Stage 2, result R3 (gap G6 - nobody tests on later data).

    python train/06_temporal_evaluation/temporal_train_eval.py
    python train/06_temporal_evaluation/temporal_train_eval.py --set temporal.cutoff=2025-07-01T00:00:00Z
    python train/06_temporal_evaluation/temporal_train_eval.py --set reference_run=experiments/runs/stage2_supervised/<R1b dir>

A random or user-disjoint split lets the model see the future: next month's editors, tag
conventions and vandalism styles are spread through training. Deployment never gets that.
This stage trains strictly on the past and tests strictly on the future.

Two refinements:

- **Authors active on both sides of the cutoff are removed from training.** Otherwise the
  model recognises the account instead of the edit, and the temporal number inherits the
  leak R1b exists to remove.
- **The test fold is broken down by month after the cutoff**, so drift shows as a slope
  rather than one averaged number that hides whether performance decays.

Pass `reference_run` (the same task under R1b) and the drop from in-distribution to future
is written into the results. Note what that drop mixes: the passage of time *and* any change
in who reviewed and how. Label-source shift is separated by stage2_evaluate.py --task; this
stage measures time.
"""

import json
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

from common import stage  # noqa: E402
from common.fit import train_stage2  # noqa: E402


def by_month(run_dir: Path, created_at: dict, min_rows: int = 50) -> list:
    from mapguard.evaluation.metrics import average_precision, roc_auc

    pred = np.load(run_dir / "predictions.npz")
    cids, y, p = pred["test_cids"], pred["test_y"], pred["test_p"]
    months = np.array([created_at.get(int(c), "")[:7] for c in cids])
    out = []
    for m in sorted(set(months)):
        sel = months == m
        if sel.sum() < min_rows or len(np.unique(y[sel])) < 2:
            out.append({"month": m, "n": int(sel.sum()), "skipped": "too few rows or one class"})
            continue
        out.append({"month": m, "n": int(sel.sum()), "positive_rate": round(float(y[sel].mean()), 4),
                    "pr_auc": round(float(average_precision(y[sel], p[sel])), 4),
                    "roc_auc": round(float(roc_auc(y[sel], p[sel])), 4)})
    return out


def main() -> int:
    args = stage.parse_args(__doc__.splitlines()[0], Path(__file__).with_name("config.json"))
    cfg, run, store, prep, masks = stage.setup(args, tag=cfg_tag(args))

    results = train_stage2(cfg, prep, masks, run)
    created = {int(c): str(t) for c, t in zip(prep.cids, prep.created_at)}
    results["test_by_month"] = by_month(run.dir, created)
    results["cutoff"] = cfg["temporal"]["cutoff"]
    results["train_window"] = [str(prep.created_at[masks["train"]].min()),
                               str(prep.created_at[masks["train"]].max())]
    results["test_window"] = [str(prep.created_at[masks["test"]].min()),
                              str(prep.created_at[masks["test"]].max())]

    ref = cfg.get("reference_run")
    if ref:
        ref_res = json.loads((Path(ref) / "results.json").read_text(encoding="utf-8"))
        ref_pr = ref_res["headline"]["test_pr_auc"][0]
        now_pr = results["headline"]["test_pr_auc"][0]
        results["drift_vs_reference"] = {
            "reference_run": ref, "reference_split": ref_res.get("split"),
            "reference_test_pr_auc": ref_pr, "temporal_test_pr_auc": now_pr,
            "change": round(now_pr - ref_pr, 4),
            "note": "mixes time drift with any change in reviewing practice over the window",
        }
    run.finish(results)
    print(json.dumps({k: results[k] for k in ("cutoff", "train_window", "test_window", "headline")},
                     indent=2))
    return 0


def cfg_tag(args) -> str:
    return "past_to_future"


if __name__ == "__main__":
    raise SystemExit(main())
