# experiments/rounds/r2_2026-10-07_altair_h100 — round 2, model v2 (2026-10-07)

Run on the SVKM Altair cluster: NVIDIA H100 PCIe, MIG slice 3g.40gb, image
`tensorflow_pbs:23.07-tf2-py3` (TensorFlow **2.12.0**, Keras 2), `mixed_float16`, batch 256.
Bundle code `8631018`, feature store schema v4 (pseudonymised, release v1.1.0).
Launched with `altair/run_round2.sh` from a Jupyter terminal. **All 8 steps exit 0. Total 37 min.**

## Layout

```
r2_2026-10-07_altair_h100/
  raw/                       as it came back - untouched (byte-for-byte checked against both zips)
    experiments/pipeline/    one log + summary JSON per run_pipeline call (8)
    experiments/runs/        53 run dirs (39 real + 14 smoke): results.json, config, provenance, training_log.csv,
                             predictions*.npz, best/model weights (*.h5, 347 MB, git-ignored)
    experiments/splits/      12 frozen splits - identical to experiments/splits in the repo
    logs/                    job.log, STATUS, timing.csv, env.txt, steps/<step>.log
    run_round2.out           console output of the background job
  analysis/
    summary.md / .json       standard tables (scripts/summarise_results.py)
    deep_dive.json           paired author-bootstrap tests, error overlap (this README §3)
    ablations_author_robust.json   ablation rows with one-per-author PR-AUC (§4)
```

The downloads (`round2_results.zip`, `experiments.zip`, `run_round2.out`) were deleted from
the repo root after extraction was verified. `experiments.zip` = the `experiments/` part of
`round2_results.zip` + `splits/`, every shared file identical, so nothing was lost.
14 `_quick` dirs are the smoke test; kept in raw, excluded from every number below.

## Timing (raw/logs/timing.csv)

| Step | Minutes |
|---|---|
| smoke (wiring check + quick chain) | 6.6 |
| main: ovid 00,01,02,07 | 4.4 |
| ablations: 8 rows × 2 seeds | 12.3 |
| ovid_modern 01,02,07,06 | 5.3 |
| T1 / T2 | 2.8 / 2.7 |
| revert_vandalism / revert | 1.5 / 1.7 |

A full deep training takes ~1–2 min on the H100 (3–8 min on the M4 CPU in round 1).

## 1. Headline (R1b, user-disjoint test)

PR-AUC; "1/author" = one changeset per (author, class), the number that cannot be inflated by
bulk-reverted accounts.

| Task | Deep v2 | Hybrid | XGBoost | Ensemble | w_deep | Deep 1/author | Hybrid 1/author |
|---|---|---|---|---|---|---|---|
| ovid | 0.866 | **0.899** | 0.860 | 0.900 | 0.15 | 0.763 | **0.832** |
| ovid_modern | 0.921 | 0.936 | 0.909 | **0.938** | 0.10 | 0.854 | 0.890 |
| T1 | 0.778 | **0.855** | 0.800 | 0.839 | 0.45 | 0.803 | **0.871** |
| T2 | 0.852 | 0.895 | 0.879 | **0.899** | 0.15 | 0.852 | 0.895 |
| revert_vandalism | 0.966 | 0.962 | 0.954 | **0.968** | 0.25 | 0.957 | 0.953 |
| revert | 0.904 | 0.912 | 0.901 | **0.917** | 0.75 | 0.902 | 0.910 |

ovid, Ovid's protocol (threshold from validation): hybrid P 0.816 R 0.805 F1 **0.811** Acc
**0.813**; ensemble F1 0.811 Acc 0.813; deep F1 0.794 Acc 0.783. Ovid (published): F1 0.8166,
Acc 0.8137.

## 2. Deep v2 vs round 1

| | Round 1 (M4) | Round 2 (H100) |
|---|---|---|
| ovid deep | 0.844 | **0.866** |
| ovid Ovid-equivalent | 0.789 | 0.827 |
| ovid_modern deep | 0.909 | 0.921 |
| T1 deep | 0.786 | 0.778 |
| T2 deep | 0.869 | 0.852 |

Not a clean comparison: the code, the feature store (v3 → v4: tokens added, hidden-version
edits dropped), TensorFlow (2.19/Keras 3/CPU/float32 → 2.12/Keras 2/GPU/float16) and precision
all changed at once. Within-round comparisons (§3, §4) are clean; across rounds only the
direction on ovid (+0.022) is worth quoting; T1/T2 are slightly lower but inside round 1's
95% intervals.

## 3. Is the deep model better than the hybrid? No.

Paired bootstrap over **authors** (2,000 resamples; an author's changesets stay together),
`deep_dive.json`:

| Task | Deep − Hybrid [95% CI] | Ensemble − Hybrid [95% CI] |
|---|---|---|
| ovid | −0.033 [−0.046, −0.021] | +0.001 [−0.002, +0.006] |
| ovid_modern | −0.016 [−0.022, −0.009] | **+0.002 [+0.001, +0.004]** |
| T1 | −0.077 [−0.106, −0.051] | **−0.016 [−0.032, −0.003]** |
| T2 | −0.043 [−0.067, −0.021] | +0.004 [−0.001, +0.009] |
| revert_vandalism | +0.003 [−0.002, +0.010] | **+0.006 [+0.003, +0.009]** |
| revert | −0.008 [−0.021, +0.005] | +0.006 [−0.004, +0.015] |

- The deep model is **significantly worse** than the hybrid on ovid, ovid_modern, T1, T2, and
  tied on the two revert tasks.
- The gap is largest one-per-author (ovid: 0.763 vs 0.832): the deep model leans more on
  repeat authors.
- The ensemble adds a small, real gain on ovid_modern and revert_vandalism, nothing measurable
  on ovid/T2/revert, and **hurts T1** (blend weight 0.45 chosen on a small validation set —
  it overfit the weight).
- Deep and hybrid rank changesets similarly (Spearman 0.78–0.90); at a 0.5 cut, 54–66% of the
  deep model's errors are also hybrid errors (64–73% the other way), so there is limited
  complementary signal to blend.

## 4. Ablations (ovid, 2 seeds each)

| Row | Branches | PR-AUC | 1/author | F1 |
|---|---|---|---|---|
| base_edits_tabular | B5 | 0.820 | 0.704 | 0.758 |
| plus_geometry | B1+B5 | 0.820 | 0.698 | 0.758 |
| plus_text | B2+B5 | 0.837 | 0.703 | 0.782 |
| plus_sequences | B3+B4+B5 | 0.830 | 0.680 | 0.766 |
| **plus_edit_meaning** | B5+B6 | **0.856** | **0.737** | 0.780 |
| full | all six | 0.863 | 0.754 | 0.798 |
| full_meanpool | all six, mean pooling | 0.861 | 0.753 | 0.792 |
| full_unweighted | all six, no author weights | 0.863 | 0.752 | 0.798 |

Seed spread is 0.0003–0.006 PR-AUC and 0.003–0.017 one-per-author.

- **B6 (edit meaning) is the one change that clearly works**: +0.036 PR-AUC and +0.033
  one-per-author over the base, the largest single-branch gain, far above seed noise. The same
  information drove the hybrid's gain.
- Geometry (B1) adds nothing (+0.0004). Sequences (B3+B4) add +0.010 PR-AUC but *lower*
  one-per-author (0.680 vs 0.704) — they help memorise repeat authors.
- **MIL vs mean pooling: no measurable difference** (0.863 vs 0.861, inside seed noise).
- **full_unweighted is not a valid test — a bug** (next section).

## 5. Bug found: author weighting never reached the deep model

`focal_loss` returned the batch **mean** (a scalar). Keras applies `sample_weight` to per-sample
losses; given a scalar it multiplies it by the batch's mean weight (~1 after normalisation), so
the weights only rescaled whole batches. Evidence: `full` and `full_unweighted` (seed 20260904)
have epoch-1 losses 0.10702 vs 0.10709 and near-identical curves throughout. The configs were
right (`author_weighting` = sqrt vs None, confirmed in each run's `config.json`); the loss
ignored them. The **hybrid** is not affected (scikit-learn/XGBoost take weights directly).

Fixed after this round (`stage2.focal_loss` now returns one loss per sample). Round-2 deep
numbers are therefore all *unweighted*; the "does weighting help the deep model?" question is
still open.

## 6. Training behaviour

- Masking check passed on every run, before and after training, under float16.
- Best epoch is mostly 3–8 (one run at 13; early stopping, patience 5). For the full model at
  its best epoch, training PR-AUC is 0.91–0.95 against validation 0.88: it still memorises
  quickly.
- Branch occlusion on the ovid model (mean |Δscore|): B5 tabular 0.103, B2 text 0.085,
  B4 user 0.059, B6 edit meaning 0.045, B1 0.040, B3 0.021.

## 7. Temporal (train past, test future)

| Task | Train → Test | PR-AUC | vs R1b |
|---|---|---|---|
| ovid_modern | 2014–19 → 2021-09…2026-06 | 0.893 | −0.028 |
| T1 | 2023-01…2025-06 → 2025-07…2026-09 | 0.782 | +0.004 |
| T2 | 2023-01…2025-06 → 2025-07…2026-09 | 0.827 | −0.025 |

A model trained only on 2014–19 still scores 0.893 on 2021–26: the signal transfers across
seven years with a modest drop.

## 8. What this round settles

1. **Best system now: the hybrid** (or the ensemble where it is significantly better). On
   ovid it matches Ovid's published F1/accuracy (0.811/0.813 vs 0.817/0.814) and is stronger
   one-per-author.
2. **Edit meaning (B6) is the key ingredient** for both model families.
3. MIL pooling is not worth claiming; geometry is not contributing.
4. The deep model's weakness is per-author generalisation, and the one tool aimed at it
   (author weighting) was silently off — the next run tests it properly.

## Not available in this round

- Per-edit attention weights were not saved, so whether MIL attention concentrates on single
  edits cannot be checked. Saved from the next run on (`test_attention` in predictions.npz).
- One seed for the headline runs (only the ablations have two).
