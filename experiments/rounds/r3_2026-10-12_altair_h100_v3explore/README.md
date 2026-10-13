# experiments/rounds/r3_2026-10-12_altair_h100_v3explore — round 3, v3 exploration

H100 MIG 3g.40gb, TF 2.12, mixed_float16, 1 CPU core, store v4 + object-context side table,
code `80dc678`, plan `altair/plans/r3_v3_explore.txt`. Finished 2026-10-12 10:08,
STATUS `FINISHED with failures: embed`.

## Layout

```
raw/experiments/   runs (two replicate copies of every step - see "what went wrong"), pipeline logs
raw/logs/          job.log, STATUS, timing.csv, env.txt, plan.normalized, steps/*.log
analysis/round3.json         every number below (round3_analysis.py)
```

The `where/` files in the zip were byte-identical to `C:/osm-data/where/` and were not duplicated.
The downloaded zip was deleted after a CRC check of all 1,131 files.

## What went wrong (identified, not yet fixed)

| # | Problem | Evidence | Effect | Fix for the next run |
|---|---|---|---|---|
| 1 | **The job ran twice at once.** Two launches ~2.5 min apart (job totals 2 h 30 m and 2 h 28 m, both ending 10:08) ran the whole plan concurrently in the same round folder. | `timing.csv`: every step listed twice with overlapping times | Both copies shared one GPU slice: steps took 3–5× longer (main 23–25 min vs 4–5 min in round 2c). Duplicate runs of everything. Concurrent pip installs into `.pydeps_st`. | Lock file per round in the job script: a second launch exits immediately with a message ("already running, PID ...", stale lock detected via PID). |
| 2 | **Text embeddings failed.** `transformers` 5.19 (latest) uses numpy typing that numpy 1.22.2 lacks: `TypeError: Too few arguments for numpy.ndarray`. The numpy pin protects TensorFlow but was also applied to the separate embedding environment, where it is not needed. | `logs/steps/embed.log` | `hybrid_emb`, `hybrid_ctx_emb`, stack `emb`, ablation rows `full_text_emb` and `full_v3` all skipped (recorded with reasons). The embedding idea is untested. | Install the embedding environment **without** the numpy pin (its own process, never imports TF), into a fresh folder (`.pydeps_st` now holds the broken numpy-1.22 install). Driver 590 / CUDA 13.1 supports the installed torch 2.14 (cu13). Add a `--quick` embed smoke on 500 rows before the full embed. |
| 3 | **Ensemble paired runs across copies.** Stage 07 takes "the newest" deep and hybrid runs; with two copies running, copy 1's ensemble used copy 2's hybrid. | `round3.json` provenance (ovid: deep `082554` + hybrid `082909`) | No numeric effect here (the hybrid is deterministic: both copies identical to 4 decimals), but provenance is unclean. | run_pipeline passes its own run dirs to stage 07 explicitly instead of "newest". |
| 4 | `scripts/summarise_results.py` crashes when the ablation set has no `base` row (`KeyError: pr_auc_change_vs_base`). | this round | Standard tables not generated; `round3.json` covers the round. | Make the "vs base" column optional. |

## Results (two replicate copies shown as copy 1 / copy 2)

PR-AUC, user-disjoint R1b test:

| Task | Hybrid | Hybrid + ctx | Deep v2.1 | Ensemble | **Stack** |
|---|---|---|---|---|---|
| ovid | 0.899 / 0.899 | 0.898 | 0.883 / 0.881 | 0.899 / 0.899 | **0.903 / 0.902** |
| ovid_modern | 0.936 | 0.934 | 0.929 / 0.926 | **0.940 / 0.941** | 0.939 / 0.939 |
| T1 | **0.855** | 0.849 | 0.793 / 0.794 | 0.845 / 0.845 | 0.847 / 0.846 |
| T2 | 0.895 | 0.891 | 0.871 / 0.874 | 0.896 / 0.897 | **0.899** / 0.895 |
| revert_vandalism | 0.962 | 0.964 | 0.965 / 0.965 | **0.968 / 0.969** | 0.966 / 0.965 |
| revert | 0.912 | 0.916 | 0.917 / 0.917 | **0.923 / 0.923** | 0.914 / 0.915 |

ovid on Ovid's protocol (F1 at the validation threshold):

| | F1 | one-per-author PR-AUC |
|---|---|---|
| Hybrid | 0.811 | 0.832 |
| Ensemble | 0.824 / 0.820 | 0.831 / 0.830 |
| **Stack** | **0.821 / 0.822** | **0.837 / 0.834** |
| Ovid (published) | 0.817 | — |

- **Stacking is the first change that lifts ovid above the hybrid**: +0.004 PR-AUC in both
  copies, F1 0.821–0.822 vs 0.811, one-per-author +0.002–0.005. This is the first time a model
  is above Ovid's published F1 (0.817). **Not yet a claim**: the gain is small; it needs the
  author-bootstrap test and multiple seeds (next round).
- **Stacking does not help T1** (0.847 vs 0.855) - the same task where the blend fails; the deep
  model is too weak there (0.793). On ovid_modern and the revert tasks the simple ensemble is
  best.
- **Object context: no gain again** (hybrid_ctx within ±0.007 of hybrid on every task), matching
  the laptop probe.
- The hybrid is deterministic (identical in both copies); the deep model varies by up to
  0.003 PR-AUC between copies of the same seed.

## Ablations (ovid; 3 seeds × 2 copies = 6 runs per row)

| Row | PR-AUC (sd) | one-per-author | F1 | same-seed replicate |diff| |
|---|---|---|---|---|
| full (v2.1) | 0.8830 (0.0015) | 0.805 | 0.809 | 0.0004, 0.0002, 0.0022 |
| full_context | 0.8793 (0.0032) | 0.799 | 0.805 | 0.0042, 0.0016, 0.0010 |
| **full_2layers** | **0.8864 (0.0008)** | **0.809** | **0.812** | 0.0010, 0.0001, 0.0010 |
| full_text_emb | skipped (embeddings failed) | | | |
| full_v3 | skipped (embeddings failed) | | | |

- **Two transformer blocks over edits: +0.0034 PR-AUC**, about 2× the seed spread of `full`,
  +0.004 one-per-author. A small but consistent gain.
- Object context in the deep model: −0.004 - not adopted.
- GPU nondeterminism: the same seed run twice differs by up to 0.004 PR-AUC, which sets the
  floor for any claimed difference between single runs.

## Next run (to do)

Fix 1–4 above, then: embeddings (hybrid `emb`, deep `full_text_emb`, `full_v3`), and the
multi-seed run (`--seed`, split frozen) of the candidates that held up here - stack on ovid/T2,
ensemble on ovid_modern/revert tasks, `edit_layers=2` - with author-bootstrap tests of each
claimed gain.
