# experiments/rounds/r4_2026-10-13_altair_h100_seeds — round 4, 3 seeds, embeddings

H100 MIG 3g.40gb, TF 2.12, mixed_float16, 1 CPU core; code `ad7cb05`, plan
`altair/plans/r4_seeds.txt`. **All 24 steps exit 0, one launch, 3 h 43 m** (2026-10-13 07:37–11:15).
Text embeddings: `paraphrase-multilingual-MiniLM-L12-v2` on CUDA (torch 2.5.1, numpy 1.22.2),
46,578 changesets × (comment, tag text) in 37 s. Every main and temporal deep run used them
(`training.text_emb_dim = 768`, `edit_layers = 2`); the Ovid reproduction stays round-1 (0, 1).
Split frozen for all seeds (`split_seed 20260904`); model seeds 20260904/05/06.

## Layout

```
raw/experiments/runs/     baselines, stage2_supervised, ensemble, stack (+5 fold runs each),
                          stage2_ablations, stage2_temporal  (1,520 files, 260 weight files git-ignored)
raw/logs/                 job.log, STATUS, timing.csv, env.txt, plan.normalized, steps/<step>.log
analysis/seeds.json       mean ± sd over 3 seeds + author bootstrap vs hybrid (scripts/analyse_seeds.py)
analysis/ablations_3seeds.json   ablation rows, 3 seeds
analysis/summary.md|json  standard tables (scripts/summarise_results.py; its ablation header says
                          "2 seeds" - the rows have 3, see ablations_3seeds.json)
```

The zip's `where/` files were byte-identical to `C:/osm-data/where/` and were not duplicated.
The zip was deleted after a CRC check of all 1,520 extracted files.

## 1. Headline — mean ± sd over 3 seeds (PR-AUC, user-disjoint R1b test)

| Task | Hybrid | Hybrid + emb | Deep v3 | Ensemble | Stack | **Stack + emb** |
|---|---|---|---|---|---|---|
| ovid | 0.8978 ±.0008 | 0.9016 ±.0001 | 0.8857 ±.0024 | 0.9024 ±.0005 | 0.9028 ±.0011 | **0.9062 ±.0012** |
| ovid_modern | 0.9359 ±.0003 | 0.9359 ±.0005 | 0.9355 ±.0032 | **0.9438 ±.0010** | 0.9405 ±.0011 | 0.9396 ±.0012 |
| T1 | 0.8520 ±.0029 | 0.8583 ±.0014 | 0.8273 ±.0051 | **0.8585 ±.0011** | 0.8561 ±.0065 | 0.8569 ±.0024 |
| T2 | 0.8953 ±.0010 | **0.8979 ±.0021** | 0.8856 ±.0072 | 0.8972 ±.0052 | 0.8967 ±.0025 | 0.8949 ±.0034 |
| revert_vandalism | 0.9629 ±.0008 | 0.9667 ±.0007 | 0.9675 ±.0016 | **0.9698 ±.0011** | 0.9654 ±.0004 | 0.9685 ±.0003 |
| revert | 0.9132 ±.0013 | 0.9162 ±.0005 | 0.9180 ±.0003 | **0.9245 ±.0010** | 0.9170 ±.0020 | 0.9202 ±.0018 |

One-per-author PR-AUC follows the same order (`seeds.json`); for ovid: hybrid 0.832,
stack + emb **0.844**, ensemble 0.836, deep 0.807.

## 2. Is the gain over the hybrid real? Author-cluster bootstrap of seed-averaged scores

Difference in PR-AUC vs the hybrid, 95% interval; **bold = interval excludes zero**:

| Task | Hybrid + emb | Deep v3 | Ensemble | Stack | Stack + emb |
|---|---|---|---|---|---|
| ovid | **+0.0037** [.0005, .0072] | −0.0074 [−.017, .002] | +0.0048 [−.001, .011] | **+0.0056** [.003, .008] | **+0.0097** [.006, .014] |
| ovid_modern | +0.0001 | +0.0032 | **+0.0086** [.005, .012] | **+0.0048** [.003, .007] | **+0.0040** [.002, .006] |
| T1 | **+0.0067** [.0004, .013] | **−0.0180** [−.035, −.0004] | +0.0083 [−.002, .019] | +0.0060 [−.001, .013] | +0.0072 [−.001, .015] |
| T2 | +0.0029 | −0.0087 | +0.0019 | +0.0024 | +0.0014 |
| revert_vandalism | **+0.0038** [.001, .007] | +0.0056 [−.000, .012] | **+0.0068** [.003, .012] | +0.0023 | **+0.0059** [.003, .010] |
| revert | +0.0033 [−.000, .007] | +0.0063 | **+0.0115** [.004, .020] | **+0.0048** [.000, .010] | **+0.0073** [.003, .012] |

- **Best per task:** stack + emb on ovid; ensemble on ovid_modern, revert_vandalism, revert;
  hybrid + emb / ensemble on T1 and T2 (no significant gain over the hybrid there).
- **Text embeddings help the trees** (significant on ovid, T1, revert_vandalism) and the stack.
- **The deep model alone** is level with or above the hybrid on the modern revert tasks
  (revert 0.918 vs 0.913) but still below it on ovid, T1, T2. Its value is as a component:
  every significant gain on ovid comes from combining it with the trees.

## 3. Against Ovid (published 2021: P 0.8035, R 0.8302, F1 0.8166, Acc 0.8137)

Seed-averaged stack + emb, threshold chosen on validation, bootstrap 95% intervals:

| | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| **Stack + emb (ours)** | 0.8008 | 0.8562 | **0.8276** [0.8144, 0.8402] | **0.8226** [0.8098, 0.8348] |
| Hybrid (ours) | 0.8060 | 0.8289 | 0.8173 [0.8030, 0.8311] | 0.8156 [0.8029, 0.8284] |
| Ovid (published) | 0.8035 | 0.8302 | 0.8166 | 0.8137 |

Our best model is **above Ovid on F1 (+1.1 points) and accuracy (+0.9)**; P(F1 > 0.8166) = 0.946
under the bootstrap, so the margin is **not conclusive at the 95% level** against Ovid's single
published number. Ovid's figure comes from its own test split, which we cannot reproduce exactly
(re-fetched changesets, hidden versions dropped), so this is a like-for-like *protocol*, not an
identical test set. Against our own hybrid on the identical test set the gain is conclusive
(+0.0097 PR-AUC, interval excludes zero).

## 4. Ablations (ovid, 3 seeds; base config v2.1)

| Row | PR-AUC | one-per-author | F1 |
|---|---|---|---|
| full (v2.1) | 0.8840 ±.0019 | 0.807 ±.005 | 0.811 |
| full_text_emb | 0.8821 ±.0021 | 0.802 ±.007 | 0.810 |
| full_2layers | 0.8857 ±.0018 | 0.807 ±.004 | 0.811 |
| full_2layers_text_emb | **0.8880 ±.0047** | **0.816 ±.013** | 0.811 |

Embeddings alone do not help the deep model (−0.002); with two transformer blocks they do
(+0.004 over full, +0.009 one-per-author) but with a larger seed spread. The 2-layer gain
(+0.0017) is smaller than in round 3 (+0.0034).

## 5. Temporal (train past, test future; one seed, deep v3)

| Task | Train → test | PR-AUC | Round 2c (deep v2.1) |
|---|---|---|---|
| ovid_modern | 2014–19 → 2021–26 | 0.874 [0.865, 0.884] | 0.908 |
| T1 | 2023-01…2025-06 → 2025-07…2026-09 | 0.806 [0.785, 0.826] | 0.793 |
| T2 | same | 0.835 [0.815, 0.855] | 0.849 |

The v3 deep model transfers **worse** across seven years (ovid_modern 0.874 vs 0.908 for v2.1):
a model trained on 2014–19 text with pretrained embeddings and a deeper edit encoder appears to
fit period-specific patterns. One seed only; to be reported as an observation, not a finding.
On the short T1/T2 horizon the change is mixed (+0.013 / −0.014).

## What this round settles

1. **Final system per task**: stack + emb (ovid), ensemble (ovid_modern, revert_vandalism,
   revert); on T1/T2 no combination beats the hybrid significantly.
2. **Text embeddings are worth keeping**, chiefly through the trees and the stack.
3. Above Ovid on its protocol (F1 0.828 vs 0.817, accuracy 0.823 vs 0.814), significant against
   our own hybrid, not conclusive against Ovid's single number.
4. Open: the temporal drop of the v3 deep model.
