# experiments/rounds/r2c_2026-10-09_altair_h100_v21 — model v2.1 (residual) + WHERE job

H100 MIG 3g.40gb, TF 2.12, mixed_float16, store v4 (pseudonymised), code `deb6f1f`.
v2.1 = v2 + residual + LayerNorm around edit self-attention, author weighting working.
All 9 steps exit 0: training 44 min, WHERE 59 min (job total 1 h 43 min).

## Layout

```
raw/experiments/      runs, pipeline logs and index for this round only (new layout: one round per zip)
raw/logs/             job.log, STATUS, timing.csv, env.txt, steps/*.log (incl. where.log)
analysis/             summary.md|json, deep_dive.json, ablations_author_robust.json, attention.json
```

The WHERE output is OSM data, not a training run: it lives in `C:/osm-data/where/`
(`where_features.parquet`, `where.log`, `source.json`, `probe_ovid.json`).

## 1. Deep model v2.1 vs v2 (r2b) - the residual pays

| Task | Deep r2b | **Deep r2c** | Deep 1/author r2c | Hybrid | Ensemble r2c |
|---|---|---|---|---|---|
| ovid | 0.873 | **0.882** | 0.802 | **0.899** | 0.898 |
| ovid_modern | 0.928 | **0.930** | 0.884 | 0.936 | **0.940** |
| T1 | 0.788 | **0.795** | 0.816 | **0.855** | 0.846 |
| T2 | 0.868 | **0.874** | 0.874 | **0.895** | 0.894 |
| revert_vandalism | 0.964 | **0.966** | 0.959 | 0.962 | **0.969** |
| revert | 0.908 | **0.915** | 0.915 | 0.912 | **0.922** |

ovid, Ovid's protocol: deep v2.1 P 0.768 R 0.863 **F1 0.813** Acc 0.803 (Ovid: F1 0.817,
Acc 0.814). The deep model alone is now at Ovid's F1.

Author bootstrap (deep − hybrid): ovid −0.016 [−0.027, −0.006], ovid_modern −0.006,
T1 −0.060, T2 −0.021 (all significant); revert_vandalism +0.003 and revert +0.004 (ties, deep
ahead). The gap halved on ovid since r2 (−0.033). Ensemble − hybrid significant on ovid_modern
(+0.004), revert_vandalism (+0.007), revert (+0.011); none elsewhere; T1 still −0.009.

## 2. Ablations (ovid, 2 seeds)

| Row | PR-AUC | 1/author | F1 |
|---|---|---|---|
| base_edits_tabular | 0.855 | 0.776 | 0.786 |
| plus_geometry | 0.857 | 0.778 | 0.786 |
| plus_text | 0.876 | 0.795 | 0.806 |
| plus_sequences | 0.851 | 0.773 | 0.784 |
| plus_edit_meaning | 0.881 | 0.804 | 0.802 |
| **full** | **0.884** | **0.806** | **0.808** |
| full_meanpool | 0.880 | 0.805 | 0.801 |
| full_unweighted | 0.868 | 0.759 | 0.802 |
| full_no_residual | 0.877 | 0.805 | 0.804 |

- Residual: +0.007 PR-AUC (full vs full_no_residual, ~2× seed spread); one-per-author unchanged.
- MIL vs mean pooling: +0.004 now (0.884 vs 0.880) - small, but it only appears once the
  residual keeps edits distinct.
- Author weighting: +0.016 PR-AUC, +0.047 one-per-author - the largest single training change.
- Edit meaning and text remain the branches that pay; geometry and sequences ~0.

## 3. Attention now picks edits (attention.json)

2,562 test changesets with ≥5 edits (median 28):

| | top weight / uniform (median) | normalised entropy |
|---|---|---|
| r2b MIL (no residual) | 1.00 | 1.000 |
| **r2c MIL (residual)** | **2.64** | 0.951 |
| r2c mean pooling | 1.30 | 0.998 |

And it is sharper on damaging changesets (2.99×) than clean ones (2.17×) - the attention is
pointing at something label-related, usable as the "which edit" explanation.

## 4. WHERE - collected, and no gain

Planet changeset dump `changesets-261005.osm.bz2` (8.81 GB, 189,976,300 changesets). 86.6 M
belong to our 24,189 authors; 46,538 / 46,578 target changesets have a bbox; 93.0% have a
distance from prior history (strictly earlier changesets only).

Probe on the hybrid (ovid, same split and code, nine WHERE columns added;
`C:/osm-data/where/probe_ovid.json`):

| | PR-AUC | 1/author | F1 | Acc |
|---|---|---|---|---|
| hybrid | 0.899 | 0.835 | 0.818 | 0.813 |
| hybrid + WHERE | 0.899 | 0.833 | 0.817 | 0.808 |

Difference −0.0005 [−0.006, +0.004] (author bootstrap): **WHERE adds nothing** on ovid. Reported
as a negative result. (This laptop rerun of the hybrid gives F1 0.818 vs 0.811 on the cluster:
same code and data, different XGBoost build - threshold sensitivity, not a real difference.)

## 5. Temporal (train past, test future)

ovid_modern 2014–19 → 2021–26: 0.908 (−0.022 vs R1b); T1 0.793 (−0.002); T2 0.849 (−0.025).
