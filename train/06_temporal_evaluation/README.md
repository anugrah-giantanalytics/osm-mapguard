# 06 · Stage 2 — temporal evaluation (R3)

**Run order 6.** Plan stage: **Stage 2**, result **R3** — gap G6. M4 or cluster.

| File | Does |
|---|---|
| `config.json` | task, cutoff date, whether to drop authors active on both sides |
| `temporal_train_eval.py` | trains on the past, tests on the future, breaks the test down by month |

Deployment only ever sees the past. Random and user-disjoint splits scatter next month's
editors and vandalism styles through training, so they overstate how a model ages.

- **Strictly before the cutoff trains, strictly after tests.**
- **Authors active on both sides are removed from training** - otherwise the model recognises
  accounts across time, the leak R1b exists to prevent.
- **Month-by-month test metrics**, so decay shows as a slope.
- `--set reference_run=<R1b run>` records the drop from in-distribution to future.

That drop mixes two things: time, and any change in how reviewers labelled over the window.
Label-*source* shift (revert-derived vs human review) is a different question, answered by
`02_stage2_supervised/stage2_evaluate.py --task`.

**Pick the cutoff after the data is final** and state it as an exact date in the paper. It
must leave enough months after it for the month breakdown to mean anything.

```bash
python train/06_temporal_evaluation/temporal_train_eval.py --dry-run
python train/06_temporal_evaluation/temporal_train_eval.py --set temporal.cutoff=2026-01-01T00:00:00Z
```

The frozen split file name includes the cutoff date, so changing the cutoff creates a new
split rather than silently reusing the old one.
