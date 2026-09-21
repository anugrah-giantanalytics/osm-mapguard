"""[ENSEMBLE · final model] Blend the deep model with the hybrid tree model.

Run order 7 (after 01 and 02). CPU only - it reads predictions, it trains nothing.

    python train/07_ensemble/ensemble.py --task ovid
    python train/07_ensemble/ensemble.py --deep <stage2 run dir> --hybrid <baselines run dir>

Why a blend. The hybrid (gradient-boosted trees over tabular features plus text and
edit-token scores) is strong on the WHO and COUNT signals; the deep model reads each edit and
its text with multiple-instance pooling. They make different mistakes, and a convex blend of
two calibrated-enough scores is the cheapest way to use both - the design large moderation
systems settle on after trying to replace trees with one network.

The blend weight is chosen on the **validation** fold only (grid 0.0-1.0), then applied once to
test. Both inputs must come from the same frozen split: rows are aligned by changeset id and
the script refuses if the id sets differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

from common import data_contract, scoring  # noqa: E402
from common.run import RUNS, Run  # noqa: E402


def newest(stage: str, task: str, exclude: str | None = None, quick: bool = False) -> Path | None:
    idx = RUNS / "index.jsonl"
    if not idx.exists():
        return None
    for line in reversed(idx.read_text(encoding="utf-8").splitlines()):
        e = json.loads(line)
        d = Path(e["run_dir"])
        d = d if d.is_absolute() else TRAIN.parent / d
        if (e["stage"] == stage and e.get("task") == task and e.get("split") == "R1b"
                and d.name.endswith("_quick") == quick and (exclude is None or exclude not in d.name)
                and d.exists()):
            return d
    return None


def load_pair(path: Path) -> dict:
    z = np.load(path)
    return {f: {int(c): float(p) for c, p in zip(z[f"{f}_cids"], z[f"{f}_p"])}
            for f in ("val", "test")} | {"y": {int(c): int(y) for f in ("val", "test")
                                               for c, y in zip(z[f"{f}_cids"], z[f"{f}_y"])}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="ovid")
    ap.add_argument("--deep", default=None, help="stage2_supervised run dir (default: newest R1b)")
    ap.add_argument("--hybrid", default=None, help="baselines run dir (default: newest R1b)")
    ap.add_argument("--feature-store", default=None)
    ap.add_argument("--quick", action="store_true", help="blend smoke-test runs (never report)")
    args = ap.parse_args()

    deep = Path(args.deep) if args.deep else newest("stage2_supervised", args.task, exclude="ovid_repro", quick=args.quick)
    hyb = Path(args.hybrid) if args.hybrid else newest("baselines", args.task, quick=args.quick)
    if deep is None or hyb is None or not (hyb / "predictions_hybrid.npz").exists():
        raise SystemExit(f"need a stage2 run and a baselines run with predictions_hybrid.npz "
                         f"for task {args.task} (found deep={deep}, hybrid={hyb})")
    d, h = load_pair(deep / "predictions.npz"), load_pair(hyb / "predictions_hybrid.npz")
    for f in ("val", "test"):
        if set(d[f]) != set(h[f]):
            raise SystemExit(f"{f} ids differ between {deep.name} and {hyb.name}: not the same split")

    store = data_contract.load(args.feature_store, task=args.task)
    uid = dict(zip(store.changesets["cid"].astype(int), store.changesets["uid"]))

    def arrays(f):
        ids = sorted(d[f])
        return (np.array([d["y"][c] for c in ids]), np.array([d[f][c] for c in ids]),
                np.array([h[f][c] for c in ids]), [uid.get(c) for c in ids])

    yv, dv, hv, _ = arrays("val")
    yt, dt, ht, gt = arrays("test")
    from mapguard.evaluation.metrics import average_precision

    grid = [round(w, 2) for w in np.linspace(0, 1, 21)]
    val_scores = {w: float(average_precision(yv, w * dv + (1 - w) * hv)) for w in grid}
    best_w = max(val_scores, key=val_scores.get)
    rep = scoring.report(yv, best_w * dv + (1 - best_w) * hv, yt, best_w * dt + (1 - best_w) * ht,
                         test_groups=gt)
    parts = {name: scoring.headline(scoring.report(yv, pv, yt, pt, test_groups=gt))
             for name, pv, pt in (("deep", dv, dt), ("hybrid", hv, ht))}

    cfg = {"stage": "ensemble", "task": args.task, "split": "R1b", "seed": 0,
           **({"tag": "quick"} if args.quick else {}),
           "deep_run": str(deep), "hybrid_run": str(hyb)}
    run = Run(cfg)
    run.finish({"headline": scoring.headline(rep), "evaluation": rep, "deep_weight": best_w,
                "validation_pr_auc_by_weight": val_scores, "components": parts})
    print(json.dumps({"deep_weight": best_w, "ensemble": scoring.headline(rep),
                      "deep": parts["deep"]["test_pr_auc"], "hybrid": parts["hybrid"]["test_pr_auc"]},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
