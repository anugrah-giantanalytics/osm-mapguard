# experiments/rounds/r2b_2026-10-08_altair_h100_weightfix — round 2 rerun with author weighting fixed

Same hardware, image, data and steps as `r2_2026-10-07_altair_h100` (H100 MIG 3g.40gb, TF 2.12,
mixed_float16, store v4). Code `8f11998`: `focal_loss` returns a per-sample loss, so author
weights now reach the deep model; per-edit attention is saved. All 8 steps exit 0, 38 min.

## Layout

```
raw/experiments/runs/       only the 2026-10-08 runs (the zip also carried the 2026-10-07 runs:
                            418 files, all byte-identical to r2_.../raw, so not duplicated here)
raw/experiments/pipeline/   the 2026-10-08 pipeline logs + summaries
raw/experiments/runs/index.jsonl   as downloaded (lists both days)
raw/logs/, raw/run_round2b.out     job logs, timing, environment
analysis/summary.md|json    standard tables
analysis/deep_dive.json     author-bootstrap tests deep / hybrid / ensemble (deep_dive.py)
analysis/ablations_author_robust.json, attention.json   (extra.py)
```

## What changed vs round 2 (deep model only - baselines and hybrid are identical)

| Task | Deep r2 | Deep r2b | Deep 1/author r2 → r2b | Hybrid | Ensemble r2b |
|---|---|---|---|---|---|
| ovid | 0.866 | **0.873** | 0.763 → **0.801** | 0.899 | 0.898 |
| ovid_modern | 0.921 | **0.928** | 0.854 → **0.884** | 0.936 | **0.941** |
| T1 | 0.778 | 0.788 | 0.803 → 0.814 | **0.855** | 0.843 |
| T2 | 0.852 | **0.868** | 0.852 → 0.868 | 0.895 | **0.902** |
| revert_vandalism | 0.966 | 0.964 | 0.957 → 0.957 | 0.962 | **0.968** |
| revert | 0.904 | 0.908 | 0.902 → 0.907 | 0.912 | **0.919** |

Author bootstrap (2,000 resamples, `deep_dive.json`): deep − hybrid is still significantly
negative on ovid (−0.025), ovid_modern (−0.008), T1 (−0.067), T2 (−0.026); tied on the two
revert tasks. Ensemble − hybrid is significantly positive on ovid_modern (+0.005) and
revert_vandalism (+0.006), not significant elsewhere, and still below the hybrid on T1.

## Ablations (ovid, 2 seeds) - weighting now real

| Row | PR-AUC | 1/author | F1 |
|---|---|---|---|
| base_edits_tabular | 0.851 | 0.768 | 0.782 |
| plus_geometry | 0.852 | 0.771 | 0.783 |
| plus_text | 0.871 | 0.783 | 0.803 |
| plus_sequences | 0.851 | 0.768 | 0.782 |
| plus_edit_meaning | 0.873 | 0.790 | 0.801 |
| **full** | **0.878** | **0.806** | **0.805** |
| full_meanpool | 0.877 | 0.806 | 0.803 |
| full_unweighted | 0.863 | 0.753 | 0.799 |

- **Author weighting works**: full vs full_unweighted +0.015 PR-AUC, **+0.053 one-per-author**.
- Edit meaning and text remain the two branches that pay; geometry and sequences add nothing.
- MIL vs mean pooling: still no difference.

## Attention (attention.json) - why MIL does nothing

Over 2,562 test changesets with ≥5 edits (median 28), the MIL weights are **exactly uniform**:
median top weight / (1/n) = 1.00, normalised entropy 1.00, no changeset with an edit above 50%.
(Under mean pooling the self-attention received is only mildly peaked: 1.83× uniform.)

Uniform gated-attention weights mean the per-edit vectors entering the pool are nearly
identical. The self-attention layer had no residual connection, so every edit was replaced by
a mixture of all edits before pooling. **v2.1** (code `677d994`) adds residual + LayerNorm and
an ablation row `full_no_residual` to test exactly this.
