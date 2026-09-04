# MapGuard

Self-supervised, geometry-aware detection of damaging edits in OpenStreetMap.

Deep Learning (702AI0C008) mini project · team of 4 · staged to week 7 / week 14.

Full architecture, evidence and phasing: **[PROJECT_PLAN.md](PROJECT_PLAN.md)**
What the plan got wrong and how it was fixed: **[docs/PLAN_REVIEW.md](docs/PLAN_REVIEW.md)**
Step-by-step build list: **[CHECKLIST.md](CHECKLIST.md)**

---

## Where things stand

| | |
|---|---|
| **Gate 0a** | FAILED - `real-changesets` S3 has zero coverage of the 2014-2019 labels (0/140 probes) |
| **Gate 0b** | PASSED - OSM API fallback verified (3/3 changesets), at **1.2 s/call** |
| **Gate 0c** | PASSED - bucket window pinned to **~147M-164M** (Feb 2024 - Mar 2025) |
| **Gate 0d** | PASSED - plan review probes. Overturned two v1 conclusions, see below |
| **Design** | Four corpora: A Ovid-era labelled, B modern bulk, C unlabelled pretrain, D live 2026 |
| **Next** | Gate 1a-1e (week 1), then Corpus A pipeline. See [CHECKLIST.md](CHECKLIST.md) |

Two things the v1 plan had wrong, both measured on 2026-09-04:

- **Corpus A geometry is not out of reach.** Overpass `adiff` returns before/after way
  geometry with resolved coordinates back to 2012, and `/api/0.6/ways?ways=<id>v<ver>`
  batches previous versions 725 at a time. The geometry branch runs on the labelled
  benchmark, not only on an unlabelled modern corpus.
- **Latest OSM data is bulk-reachable.** The minute and changeset replication streams are
  live to the current minute, free and account-free - so the temporal test is a real 2026
  evaluation rather than a demo.

Raw gate output: [`results/`](results/)

## Layout

```
data/labels/      ovid_labels.tsv — 18,276 changesets, 9,138 vandalism (tracked)
data/raw/         fetched changeset contents (gitignored)
data/interim/     featurised Parquet (gitignored)
scripts/          one-shot investigations; gate0*.py are the data-availability gates
src/mapguard/     library code
notebooks/        thin Colab/Kaggle notebooks — git pull, import, run
results/          gate output and experiment metrics (tracked)
```

## Setup

**TensorFlow does not support Python 3.14.** Use 3.12:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Local Python is only needed for baselines and figures. Model training runs on Colab (iteration) and Kaggle (pretraining, ablation grid) — see PROJECT_PLAN.md §9.

## Rules

- **No data in this repo, and none in OneDrive.** Bulk data goes in `C:\osm-data\`.
- Notebooks stay thin: `git pull`, `pip install -r requirements.txt`, import from `src/`. No long cells.
- Checkpoint to Drive every epoch — Colab kills sessions at 12 h and after 90 min idle.
- Metrics go to `results/*.json`, not stdout.
- Be polite to the OSM API: 1 req/s, cache to disk, resume on restart, back off on HTTP 509.

## Reproducing the gates

```bash
python scripts/gate0_coverage.py --sample 150   # bucket coverage of Ovid labels
python scripts/gate0b_fallback.py               # API fallback + coverage ladder
python scripts/gate0c_bounds.py                 # exact window + previous-version endpoint
```

## Workstreams

| | Owner | Scope |
|---|---|---|
| WS1 | | data pipeline — fetchers, Parquet, DuckDB feature store, user histories |
| WS2 | | baselines + R1–R4 evaluation harness, calibration |
| WS3 | | Stage 2 branches B1–B5, attention fusion, training loop |
| WS4 | | Stage 1 pretext tasks, corruption generator, demo, figures |

WS1 gates the others, so WS2–4 build against a small mock Parquet in week 1 rather than waiting.
