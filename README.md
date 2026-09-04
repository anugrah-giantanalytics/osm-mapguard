# MapGuard

Self-supervised, geometry-aware detection of damaging edits in OpenStreetMap.

Deep Learning (702AI0C008) mini project · team of 4 · staged to week 7 / week 14.

Full architecture, evidence and phasing: **[PROJECT_PLAN.md](PROJECT_PLAN.md)**
What the plan got wrong and how it was fixed: **[docs/PLAN_REVIEW.md](docs/PLAN_REVIEW.md)**
Step-by-step build list: **[CHECKLIST.md](CHECKLIST.md)**

---

## Where things stand

**Four-week build: 2026-09-04 to 2026-10-02.** Scope cut from the original 15-week plan;
what went and why is in [PROJECT_PLAN.md](PROJECT_PLAN.md) section 13.

| | |
|---|---|
| **Gate 0** | bucket has zero coverage of the 2014-2019 labels; OSM API fallback verified |
| **Gate 0d** | plan review probes - overturned two v1 conclusions |
| **Gate 1a** | **PASS** - 93/100 geometry yield, 2,109/2,109 ways with coordinates |
| **Labels** | OSMCha token live; 3,200+ human-reviewed harmful changesets for 2026 |
| **Corpora** | A (Ovid-era labelled) - C (unlabelled pretrain) - D (live 2026). B dropped |
| **Next** | Gates 1b/1c/1d, then the feature store. See [CHECKLIST.md](CHECKLIST.md) |

Two things the original plan had wrong, both measured on 2026-09-04:

- **Corpus A geometry is not out of reach.** Overpass `adiff` returns before/after way
  geometry with resolved coordinates back to 2012, and `/api/0.6/ways?ways=<id>v<ver>`
  batches previous versions 725 at a time. The geometry branch runs on the labelled
  benchmark, not only on an unlabelled modern corpus.
- **Latest OSM data is bulk-reachable.** The minute and changeset replication streams are
  live to the current minute, free and account-free - so the temporal test is a real 2026
  evaluation rather than a demo.

Raw evidence: [`results/`](results/)

## Layout

```
data/labels/      ovid_labels.tsv - 18,276 changesets, 9,138 vandalism (tracked)
data/mock/        synthetic corpus in the real schema (generated, gitignored)
docs/             plan review, architecture page
scripts/          gates and one-shot jobs; each writes JSON to results/
src/mapguard/
  config.py       paths, endpoints, User-Agent, politeness defaults
  data/
    http.py       PoliteSession - global token bucket, 509/429 backoff
    osm_api.py    changeset metadata, osmChange download, versioned multi-fetch
    overpass.py   adiff - before/after geometry for any date from 2012
    replication.py  the live changeset stream (Corpus D, revert mining)
    osmchange.py  osmChange XML -> per-edit records, streaming
    users.py      account_created cache, account age at edit time
    store.py      resumable SQLite stores
  features/
    geometry.py   displacement, length, area, Hausdorff, changeset summary
    build_corpus_a.py  SQLite -> Parquet
tests/            38 tests, no network
results/          gate output and experiment metrics (tracked)
notebooks/        thin Colab/Kaggle notebooks - git pull, import, run
```

Bulk data lives under `MAPGUARD_DATA` (default `C:\osm-data`), never in the repository.

## Setup

**TensorFlow does not support Python 3.14.** Use 3.12 for anything that trains:

```bash
py -3.12 -m venv .venv
```

```bash
pip install -r requirements.txt && pip install -e .
```

```bash
cp .env.example .env
```

Data collection, the gates and the tests need only `requests`, `numpy`, `scikit-learn`
and `pyarrow`, so they run on the system Python too.

## Running things

Gates, in order. Each writes its evidence to `results/`:

```bash
python scripts/gate1a_adiff_batch.py --sample 100
```

```bash
python scripts/gate1b_overpass_capacity.py --minutes 30
```

```bash
python scripts/fetch_corpus_a_meta.py --rate 3 --workers 3
```

```bash
python scripts/gate1c_b0_tripwire.py
```

Unblock WS2/3/4 without waiting for the fetch - a synthetic corpus in the real schema:

```bash
python scripts/make_mock_corpus.py
```

Tests:

```bash
python -m pytest -q
```

## Rules

Full version in [CONTRIBUTING.md](CONTRIBUTING.md). The short one:

- **No data and no secrets in this repository.** Bulk data goes in `C:\osm-data\`.
- Notebooks stay thin: `git pull`, install, import from `src/`, run.
- Checkpoint to Drive every epoch - Colab kills sessions at 12 h and after 90 min idle.
- Metrics go to `results/*.json`, not stdout.
- Be polite to OSM: one shared token bucket, real User-Agent, back off on 509, resume
  from cache. Never send Overpass a planet-sized bbox.
- Three rules that are correctness, not style: user features are point-in-time, splits
  are user-disjoint, pretraining excludes the evaluation set by id **and** uid.

## Workstreams

| | Owner | Scope |
|---|---|---|
| WS1 | | data pipeline - fetchers, adiff enrichment, Parquet, DuckDB store, user histories |
| WS2 | | baselines + R1-R4 evaluation harness, calibration, label audit |
| WS3 | | Stage 2 branches B1-B5, hierarchical encoder, attention fusion, training loop |
| WS4 | | Stage 1 pretext tasks, corruption generator, demo, figures |

WS1 gates the others, so WS2-4 build against `data/mock/` in week 1 rather than waiting.
