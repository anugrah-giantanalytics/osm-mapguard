# MapGuard v3 — what changes after rounds 2–2c, and how each change is tested

Status 2026-10-13: round 3 run (2026-10-12); round 4 (embeddings + 3 seeds) built. Round 3 (`altair/plans/r3_v3_explore.txt`)
measures every item below against v2.1 on the same frozen split. Nothing here is a result until
that round reports it. Round evidence: `experiments/rounds/r2*/README.md`.

## Where v2.1 stands (round 2c, ovid, user-disjoint test)

| Model | PR-AUC | one-per-author | F1 | Acc |
|---|---|---|---|---|
| Hybrid (trees + edit tokens + text) | 0.899 | 0.832 | 0.811 | 0.813 |
| Deep v2.1 | 0.882 | 0.802 | 0.813 | 0.803 |
| Ovid (published) | — | — | 0.817 | 0.814 |

Level with Ovid, not above it: Ovid's F1 is inside our 95% interval. The ablations say what
pays: **edit meaning** (B6) and **text** (B2). Geometry and edit/user sequences add ~0.

## The v3 candidates

Each is a config switch (default off), so v2.1 is always reproducible and every comparison
is one switch on the same split.

| # | Change | Why (evidence) | Switch | Where |
|---|---|---|---|---|
| 1 | **Object context**: what each modified/deleted object *was* - had wikidata/wikipedia, name, historic, tourism, ...; tokens like `@historic`, `m@name`, `@tourism=museum` | Edit meaning is the strongest signal; it says *what changed*, not *what it was done to*. Vandals target prominent objects. | hybrid variant `ctx`; deep `model.context_tabular` | `scripts/probes/object_context_extract.py` → `object_context.parquet` |
| 2 | **Pretrained multilingual sentence embeddings** of comment and tag text (`paraphrase-multilingual-MiniLM-L12-v2`, frozen) | Text pays (+0.02 in every ablation); the char-CNN learns from 18k rows, OSM text is multilingual | hybrid variant `emb`; deep `model.text_emb` | `train/08_text_embeddings/embed_text.py` (cluster) |
| 3 | **Transformer blocks over edits** (feed-forward + self-attention, residual + LayerNorm) | The residual (v2.1) made MIL attention work (+0.007); a full block is the next step | `model.edit_layers=2` | `src/mapguard/models/stage2.py` |
| 4 | **Stacking**: deep model trained on 5 author folds; its out-of-fold score becomes a tree feature | The stage-07 blend overfits small validation sets (T1: ensemble −0.009 vs hybrid). A tree can learn *when* to trust the deep score. | stage `09` | `train/09_stack/stack_train.py` |
| 5 | **Bagged blend weight** for stage 07: mean of the best weight over 200 author-bootstrap resamples of validation | Measured on round-2c predictions: never worse than ±0.0007, T2 +0.003; does **not** fix T1 (the deep model is too weak there - item 4 addresses it) | default in stage 07 | `train/07_ensemble/ensemble.py` |
| — | WHERE (distance from the author's usual area) | Collected from the planet changeset dump; probe on the ovid hybrid: −0.0005 [−0.006, +0.004]. **Negative result, not pursued.** | hybrid variant `where` | `altair/where_job.py` |


### Measured on the laptop, 2026-10-11 (hybrid, R1b, `results/probes_v3_2026-10-11.json`)

| Task | hybrid | + object context | + WHERE |
|---|---|---|---|
| ovid | 0.899 | 0.897 | 0.899 |
| ovid_modern | 0.936 | 0.935 | — |
| T1 | 0.851 | 0.846 | — |
| revert | 0.915 | 0.918 | — |

Object context: all differences inside the 95% intervals - **no gain for the trees**, likely
because the edit tokens already carry most of it (`-name`, `~historic`). It is real signal
(damaging changesets touch objects with a Wikidata link more often: corpus R 17.2% vs 3.7%,
OSMCha 9.4% vs 4.8%; coverage of the previous version is 99.3–99.9% in both classes, so no
fetch bias), just redundant. Kept as one deep ablation row (`full_context`); dropped from
stacking.

Tree tuning (6 settings, 3 author-grouped CV folds on training rows only): ovid 0.8997 vs
0.8990, one-per-author 0.838 vs 0.835 - no meaningful gain; tree settings are not the
bottleneck. Kept as an option (`hybrid_variants: tuned`), not run by default.

**Reading.** On Ovid's benchmark the hybrid sits at ~0.90 whatever cheap signal is added
(WHERE, object context, tuning). The likely ceiling is label noise: a revert is not always
vandalism. The remaining candidates (embeddings, transformer blocks, stacking) are measured
in round 3.


## Round 3 results (2026-10-12) and the round-4 decision

Full report: `experiments/rounds/r3_2026-10-12_altair_h100_v3explore/README.md`.

| Change | Round 3 | Decision |
|---|---|---|
| Stacking (stage 09) | ovid 0.903 vs hybrid 0.899, F1 0.821 vs 0.811 (Ovid 0.817); T1 worse (0.847 vs 0.855) | candidate - confirm with 3 seeds + author bootstrap |
| `edit_layers=2` | +0.0034 PR-AUC (~2× seed sd), +0.004 one-per-author | **adopted** as the deep default |
| Object context (deep) | −0.004 | dropped |
| Object context (hybrid) | within ±0.007 on all tasks | dropped |
| Text embeddings | not tested - install failed (numpy conflict) | round 4, fixed environment |
| Ensemble | best on ovid_modern and both revert tasks | kept, explicit run pairing |

GPU nondeterminism measured: the same seed twice differs by up to 0.004 PR-AUC.

Round 4 (`altair/plans/r4_seeds.txt`): embeddings, ablation rows full / full_2layers /
full_text_emb / full_2layers_text_emb × 3 seeds, then every task × 3 seeds (frozen split) with
hybrid (base, emb), deep v3 (edit_layers 2, text_emb auto), ensemble and stack, plus temporal
evaluation. The deep model uses the embeddings only if stage 08 succeeds; whether it did is
recorded in every run (`training.text_emb_dim`).

## Leakage rules (unchanged, extended)

- Object context uses only the *previous* version of each object (what existed before the
  edit); the same source as `diff_tokens`.
- Embeddings are label-free: every row is embedded identically, nothing is fitted.
- Stacking: a deep score is never a feature for a row its model trained on. Validation/test
  rows get the mean of the 5 fold models, none of which saw them.
- `split_seed` is now separate from `seed`: multi-seed runs vary initialisation only, never
  the test set (before this, changing `seed` would have drawn a new split).

## How round 3 decides

On ovid, 3 seeds per row: `full` (v2.1) vs `full_text_emb`, `full_context`, `full_2layers`,
`full_v3`. On all six tasks: hybrid `base / ctx / emb / ctx_emb`, deep v2.1, bagged ensemble,
and stack. A change is adopted only if its gain over v2.1 / base exceeds the seed spread and
the author-bootstrap interval excludes zero. The final round then runs the adopted setup with
5 seeds.
