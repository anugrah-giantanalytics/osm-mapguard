# MapGuard

Self-supervised, geometry-aware detection of damaging edits in OpenStreetMap.

Deep Learning (702AI0C008) mini project · team of 4 · four-week build to 2026-10-02.

| | |
|---|---|
| Plan | [PROJECT_PLAN.md](PROJECT_PLAN.md) |
| What the plan got wrong | [docs/PLAN_REVIEW.md](docs/PLAN_REVIEW.md) |
| Build checklist | [CHECKLIST_v2.md](CHECKLIST_v2.md) |
| Fetch -> build -> bundle -> train | [RUNBOOK.md](RUNBOOK.md) |
| GPU cluster | [RUN_ALTAIR.md](RUN_ALTAIR.md) |

---

## Where things stand

Every row below is backed by a committed file in [`results/`](results/). Nothing is
claimed that has no artefact.

| Gate | Verdict | Evidence |
|---|---|---|
| **0a** bucket coverage of Ovid labels | FAIL — 0/140, all 404 | `gate0_coverage.json` |
| **0b/0c** OSM API fallback, bucket bounds | PASS | `gate0b_fallback.json`, `gate0c_bounds.json` |
| **0d** plan-review probes | PASS — overturned two v1 conclusions | `gate0d_review_probes.json` |
| **1a** adiff geometry yield | **PASS** — 93/100, 2,109/2,109 ways with coordinates | `gate1a_adiff_batch.json` |
| **1b** Overpass capacity | **SUBSET** — 3.17 q/min, 96 h for full corpus | `gate1b_overpass_capacity.json`, [decision](docs/GATE1B_DECISION.md) |
| **1c** label-leakage tripwire | **PASS** — AUC 0.656, weak prior not a leak | `gate1c_b0_tripwire.json` |
| **1d** framework spike | **PASS** — masking delta 0.0 (Keras 3, local) | `gate1d_framework_spike.json` |
| **1d** re-verify on cluster Keras 2 | not yet run | `cluster/00_env_check` |

### Data (final, 2026-09-19)

Sources and why: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). Coverage by label:
`results/feature_store_build_report.json`. Every column: `results/feature_schema.json`.

**Feature store: 46,578 changesets · 2.05 M sampled edits · 3.15 M point-in-time history
steps · 100% history coverage.** All raw sources fetched; nothing partial.

| Corpus | Years | Label | Changesets (neg / pos) |
|---|---|---|---|
| **A** Ovid benchmark | 2014–2019 | reverted | 8,906 / 9,131 |
| **R** revert-mined from the planet changeset dump (78.4 M changesets scanned) | 2021-09 → 2026-06 | reverted by another account ≤ 90 days; Ovid's "mentions vandalism" flag | 8,514 / 8,204 |
| **M** OSMCha | 2023 → 2026 | human review (harmful vs flagged-and-cleared) | 3,945 / 3,947 |
| **N** random | 2023 → 2026 | presumed clean | 3,931 / — |

Leaks found in the real data and removed before any training: `comments_count`, account age
(deleted accounts), empty changesets, and edits whose previous version was hidden by
moderators (R positives 4.0% vs negatives 1.9%).

### Baselines (user-disjoint R1b = Ovid's protocol; `results/baselines_*.json`)

| Task | Rows | B0 tripwire ROC-AUC | LogReg PR-AUC | XGBoost PR-AUC |
|---|---|---|---|---|
| `ovid` (A) | 18,037 | 0.661 | 0.808 | 0.860 |
| `revert` (R) | 11,280 | 0.731 | 0.895 | 0.901 |
| `revert_vandalism` (R, Ovid rule) | 7,080 | 0.839 ⚠ | 0.949 | 0.954 |
| `ovid_modern` (A + R vandalism) | 25,117 | 0.734 | 0.863 | 0.910 |
| `T1` triage (M) | 7,892 | 0.693 | 0.771 | 0.802 |
| `T2` detection (M + N) | 7,878 | 0.667 | 0.860 | 0.877 |

⚠ On `revert_vandalism`, metadata alone (new account, short comment, editor) nearly separates
the classes. That is a property of Ovid's label, not a pipeline leak - report it beside every
number on that task. On `ovid`, XGBoost reaches F1 77.7 / accuracy 76.2 against Ovid's published
81.66 / 81.37 on the same protocol; the neural model has to beat 81.66.

TensorFlow is verified on Keras 3.15 and Keras 2.13 (the cluster's version): wiring check PASS,
and the full pipeline runs end to end on the real store.

### Two things the original plan had wrong

Both measured, both recorded in `results/gate0d_review_probes.json`:

- **Corpus A geometry is reachable.** Overpass `adiff` returns before/after way geometry
  with resolved coordinates back to 2012, and `/api/0.6/ways?ways=<id>v<ver>` batches
  previous versions 725 at a time. The geometry branch runs on the published benchmark,
  not only on an unlabelled modern corpus.
- **Latest OSM data is bulk-reachable.** The minute and changeset replication streams are
  live to the current minute, free and account-free — so the temporal test is a real 2026
  evaluation rather than a demo.

---

## Layout

```
src/mapguard/          library - everything real lives here
  config.py            paths from MAPGUARD_DATA, endpoints, rate defaults, subset rule
  data/                http (global token bucket), osm_api, overpass (adiff),
                       replication, osmcha, osmchange parser, users, store
  features/            geometry, user_history (point-in-time), build_store
  evaluation/          splits (R1a/R1b), metrics (prevalence-corrected)
  models/              stage2 - five branches, masked attention, Keras 2/3 compatible
  training/            dataset assembly (padding + masking)
scripts/               local CLI - fetchers, gates, fixtures. Network-bound, no GPU.
train/                 every training and evaluation entry point, one folder per stage,
                       named by run order and tagged with plan stage - see train/README.md
  run_pipeline.py      one command for every stage, on any machine
  setup_env.sh         M4 / Linux training environment + wiring check
cluster/               one generated notebook per train/ stage, plus 99_full_pipeline
  make_notebooks.py    generates them
  bundle.py            code + feature store + splits -> one zip for cluster or M4
tests/                 111 tests, no network
data/labels/           ovid_labels.tsv - 18,276 changesets (tracked)
data/fixtures/         real changesets selected as awkward cases (tracked, small)
results/               gate output and metrics (tracked)
experiments/           checkpoints and run outputs (gitignored)
docs/                  plan review, Gate 1b decision, schema
```

Bulk data lives under `MAPGUARD_DATA` (default `C:\osm-data`), never in the repository.

## Setup

**Supported interpreter: Python 3.11 or 3.12.** TensorFlow does not support 3.13+, and the
development machine's 3.14 can run everything except model code. There is no checked-in
venv — create your own:

```bash
py -3.12 -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt && pip install -e .
```

```bash
cp .env.example .env
```

The fetchers, gates and tests need only `requests`, `numpy`, `scikit-learn` and `pyarrow`,
so they run on 3.14 too. Model code runs on the cluster.

## Running things

Data collection (local, network-bound):

```bash
python scripts/fetch_corpus_a_meta.py --rate 3 --workers 3
```

```bash
python -m mapguard.data.fetch_corpus_a --rate 3 --workers 3
```

```bash
python scripts/fetch_corpus_a_geometry.py
```

```bash
python -m mapguard.features.build_store
```

Tests:

```bash
python -m pytest -q
```

Cluster work — see [RUN_ALTAIR.md](RUN_ALTAIR.md):

```bash
python cluster/bundle.py --task 00_env_check
```

## Rules

Full version in [CONTRIBUTING.md](CONTRIBUTING.md). The short one:

- **No data and no secrets in this repository.**
- Notebooks are generated by `cluster/make_notebooks.py`, not hand-written.
- Metrics go to `results/*.json` with config, split, seed, checkpoint id and git SHA.
- Be polite to OSM: one shared token bucket, real User-Agent, back off on 509, resume from
  cache. Never send Overpass a planet-sized bbox. **Never fetch from the cluster.**
- Three rules that are correctness, not style: user features are point-in-time, splits are
  user-disjoint *and* class-stratified, pretraining excludes the evaluation set by id and uid.

## Workstreams

| | Owner | Scope |
|---|---|---|
| WS1 | | data pipeline — fetchers, adiff enrichment, feature store, user histories |
| WS2 | | baselines, R1–R4 harness, calibration, label audit |
| WS3 | | Stage 2 branches, fusion, training loop |
| WS4 | | Stage 1 pretext task, figures, demo |
