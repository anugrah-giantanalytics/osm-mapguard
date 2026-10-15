# MapGuard

**Detecting damaging edits to OpenStreetMap.**

OpenStreetMap is edited by millions of volunteers. A small share of edits is damaging: fake parks
drawn to spawn game items, invented roads, deleted landmarks, abuse written into names.
Volunteer reviewers catch these by hand, using rule-based flags. MapGuard learns from edits
that were reverted or rejected, and scores every *changeset* (one upload of edits) for how
likely it is to be damaging, with an explanation of why.

---

## Results

All numbers are measured on a **user-disjoint test set**: no author in the test set appears in
training. Each is the mean of 3 seeds, and the intervals are 95% author-cluster bootstrap
intervals. Every number traces back to a file in [`experiments/rounds/`](experiments/rounds/).

### Against Ovid (the published model for this task, 2021)

Same benchmark labels (changesets later reverted, 2014–2019) and the same protocol
(user-disjoint split, threshold chosen on validation):

| | Precision | Recall | **F1** | **Accuracy** |
|---|---|---|---|---|
| Ovid (published) | 0.804 | 0.830 | 0.817 | 0.814 |
| **MapGuard (stack + embeddings)** | 0.801 | 0.856 | **0.828** [0.814, 0.840] | **0.823** [0.810, 0.835] |

MapGuard is **+1.1 F1 and +0.9 accuracy points above Ovid**. Ovid's figure falls just inside our
interval (P(ours higher) = 0.95), so this is a gain on the same protocol, not proof on an
identical test set: the changesets were re-fetched, and edits that OSM has since hidden were
dropped.

### Every task (PR-AUC, the area under the precision–recall curve; 1.0 is perfect)

| Task | Labels | Best model | **MapGuard** | Tree baseline | Gain [95% CI] |
|---|---|---|---|---|---|
| **ovid** | reverted, 2014–19 (Ovid's benchmark) | stack + emb | **0.906** | 0.898 | **+0.010** [0.006, 0.014] |
| ovid_modern | Ovid's rule (revert mentions vandalism) on 2014–19 + 2021–26 together | ensemble | **0.944** | 0.936 | **+0.009** [0.005, 0.012] |
| revert_vandalism | reverted with "vandalism" in the revert, 2021–26 | ensemble | **0.970** | 0.963 | **+0.007** [0.003, 0.012] |
| revert | reverted by another account, 2021–26 | ensemble | **0.925** | 0.913 | **+0.012** [0.004, 0.020] |
| T1 | OSMCha human review, harmful vs flagged-and-cleared | ensemble | 0.859 | 0.852 | +0.008 [−0.002, 0.019] |
| T2 | OSMCha harmful vs random edits | hybrid + emb | 0.898 | 0.895 | +0.003 [−0.002, 0.008] |

The tree baseline is XGBoost on account, count and text features, plus out-of-fold token and
text scores, which is already stronger than Ovid. A gain in **bold** has a 95% interval that
excludes zero.

**The stricter score.** In Ovid's labels, 56 accounts supply 37% of the damaging changesets, so
a model can score well by recognising a few accounts. Counting **one changeset per author**, a
plain XGBoost drops from 0.86 to 0.77. MapGuard's best ovid model keeps **0.844**. Every result
in this repository reports this score alongside the standard one.

---

## How it works

A changeset is judged on three kinds of evidence, which are complementary: each alone scores
0.60–0.63 under the one-per-author measure.

| | Question | Signals |
|---|---|---|
| **Who** | Does this account know the map? | account history, edit rate, editor |
| **What** | What did each edit actually do? | edit tokens: `+leisure=park`, `-historic`, `~highway:track>motorway` |
| **Text** | What did they write? | comment and tag text, plus pretrained multilingual sentence embeddings |

The edit tokens are the single most useful signal. Ranked by a linear model, the most damaging
are `+leisure=park`, `+highway=footway`, `+tourism`, `+highway=motorway` and `-historic`: the
documented Pokémon-Go vandalism and fake roads. The most benign are `+source`, `+access=private`
and `+highway=track`, the habits of careful mappers.

**Models**

```
                 ┌─ hybrid: gradient-boosted trees on account/count features
changeset ───────┤          + out-of-fold logistic scores of edit tokens, text, embeddings
                 │
                 └─ deep: six-branch neural network
                      edits ──► per-edit encoder (numbers + tokens + old→new text)
                            ──► 2 transformer blocks over edits
                            ──► gated-attention multiple-instance pooling + max
                      comment/tags ──► character CNN + sentence embeddings
                      account history ──► GRU        account/counts ──► MLP
                                                │
              stack: the deep model's out-of-fold score becomes a feature of the trees
              ensemble: blend of deep + hybrid, weight chosen on validation
```

- **Multiple-instance pooling.** One bad edit among 200 good ones makes a changeset bad, so the
  model learns attention over edits rather than averaging them. With the residual connection,
  the attention concentrates on one edit (2.6× uniform, sharper on damaging changesets). That
  edit is the "which edit" explanation.
- **Author-balanced training.** Each changeset is weighted 1/√(changesets by its author). This
  added +0.05 to the one-per-author score of the deep model.
- **Stacking without leakage.** The deep model is trained on 5 author folds. Every training
  changeset gets a score from a model that never saw it, and validation and test get the mean
  of the fold models.

### What did *not* help (measured, kept as negative results)

| Idea | Result |
|---|---|
| Edit geometry (shape and area change) | ≈ +0.001 in every ablation (within seed noise) |
| Where the author usually maps (distance to past edits, from the full planet changeset history) | −0.0005 [−0.006, +0.004] |
| What the edited object was (Wikidata link, name, historic, …) | within ±0.007 on every task: real signal, but redundant with the edit tokens |
| Tuning the trees | +0.0007 |
| Self-supervised pretraining of the edit encoder | ±0.003 |

### Limits, measured

- **Revert labels are noisy.** In an [error analysis](experiments/rounds/r4_2026-10-13_altair_h100_seeds/analysis/error_analysis.md)
  of the 60 most confident mistakes:
  - the missed "damaging" changesets are mostly good-faith work reverted for other reasons
    (mapathon imports, bulk buildings, edit wars, technical tag disputes);
  - the false alarms are mostly newcomers adding real parks and footpaths, plus at least two
    that look like unreverted vandalism.
- **Over time.** Trained on 2014–19 and tested on 2021–26, the tree model holds (0.905). The
  deepest neural variant drops more (0.874, one seed), so the combined models are the robust
  choice.
- **Prevalence.** Real vandalism is well under 1% of edits. Every score above is on balanced
  data; the reports also give precision at realistic prevalence.

---

## Data

**46,578 changesets** from four sources:

| Corpus | Period | Label | Rows |
|---|---|---|---|
| A | 2014–2019 | Ovid's benchmark: later reverted | 18,037 |
| R | 2021-09 → 2026-06 | reverted by another account within 90 days, mined from the planet changeset dump | 16,718 |
| M | 2023 → 2026 | OSMCha human review (harmful vs flagged-and-cleared) | 7,892 |
| N | 2023 → 2026 | random changesets, presumed clean | 3,931 |

Each changeset carries its edits, with previous versions recovered for 99%+ of modified
objects, and the author's history *before* that changeset only. User ids are pseudonymised
(salted hash) in every released file, and usernames are removed.

- **Download:** [GitHub releases](../../releases) (feature store, frozen splits, dataset card).
- **Sources and coverage:** [docs/data/DATA_SOURCES.md](docs/data/DATA_SOURCES.md),
  [dataset card](docs/data/DATASET_CARD.md).
- **Licence:** data is a derivative of OpenStreetMap, **ODbL-1.0, © OpenStreetMap contributors**.

---

## Quick start

Python 3.11/3.12 (TensorFlow has no Python 3.13+ wheels).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q
```

Train and evaluate everything for one task on the released feature store:

```bash
python train/run_pipeline.py --feature-store <path-to>/feature_store --task ovid
```

`--task` is one of `ovid`, `ovid_modern`, `revert`, `revert_vandalism`, `T1`, `T2`. Add `--quick`
for a smoke test, and `--seed N` for another seed: the test split stays frozen. Stages:

| Stage | What |
|---|---|
| 00 | wiring check: the model builds, masks padding and trains on this machine |
| 08 | pretrained multilingual sentence embeddings |
| 01 | baselines: metadata tripwire, logistic regression, XGBoost, hybrid |
| 02 | deep model |
| 07 | ensemble |
| 09 | stacking |
| 03 | ablations |
| 06 | temporal evaluation (train past, test future) |

To rebuild the data from OpenStreetMap, or train on a GPU cluster, see
[docs/running/RUNBOOK.md](docs/running/RUNBOOK.md) and [altair/README.md](altair/README.md).

---

## Repository

```
src/mapguard/        data fetchers, feature builders, the model (models/stage2.py), metrics
train/               one folder per stage + run_pipeline.py
scripts/             data harvesting, packaging, result summaries, probes
altair/              GPU-cluster job script and per-round step plans
cluster/             notebook versions of the stages
experiments/rounds/  every training round: raw outputs, analysis, README with findings
experiments/splits/  the frozen train/validation/test splits
results/             data-gate measurements, baselines, probes
docs/                model / data / running / planning documents (index: docs/README.md)
tests/               test suite
```

Each round folder in `experiments/rounds/` documents one training run end to end: what changed,
what went wrong, every number and its source. Read them in order (r1 → r4) to see how the
model got here.
