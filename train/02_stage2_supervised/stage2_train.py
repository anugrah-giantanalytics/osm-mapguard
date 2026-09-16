"""[STAGE 2 · supervised] Train the five-branch changeset classifier.

Run order 2 of 6. Plan stage: Stage 2. Produces the headline R1 result for one task.

    python train/02_stage2_supervised/stage2_train.py
    python train/02_stage2_supervised/stage2_train.py --set task=T1
    python train/02_stage2_supervised/stage2_train.py --set split=R1a --set tag=ovid_protocol
    python train/02_stage2_supervised/stage2_train.py --dry-run

R1a (random split) is Ovid's protocol and is user-leaky; it exists only for comparison
with published numbers. R1b (user-disjoint) is the honest result. Run both, report both,
and label R1a as leaky wherever it appears.

Writes experiments/runs/stage2_supervised/<timestamp>_<task>_<split>/.
"""

import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN.parent / "src")]

from common import stage  # noqa: E402
from common.fit import train_stage2  # noqa: E402


def main() -> int:
    args = stage.parse_args(__doc__.splitlines()[0], Path(__file__).with_name("config.json"))
    cfg, run, store, prep, masks = stage.setup(args)

    results = train_stage2(cfg, prep, masks, run)
    results["split_leaky"] = cfg["split"] == "R1a"
    results["data_missing_optional_columns"] = store.missing_optional
    run.finish(results)

    h = results["headline"]
    print(f"\n{cfg['task']} / {cfg['split']}"
          f"{'  (LEAKY - comparison only)' if results['split_leaky'] else ''}")
    print(f"  test PR-AUC  {h['test_pr_auc'][0]}  95% CI [{h['test_pr_auc'][1]}, {h['test_pr_auc'][2]}]")
    print(f"  test ROC-AUC {h['test_roc_auc'][0]}")
    print(f"  PR-AUC at 1% prevalence {h['test_pr_auc_at_1pct_prevalence']}")
    print(f"  ECE {h['test_ece']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
