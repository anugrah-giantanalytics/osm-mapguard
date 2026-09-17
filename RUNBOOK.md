# MapGuard runbook — from raw OSM data to trained models

Three places, three jobs. Nothing crosses the lines below.

| Where | Does | Never does |
|---|---|---|
| **Laptop, on a good connection** | fetch, build the feature store, bundle | train a neural model |
| **SVKM cluster** (`cluster/*/*.ipynb`) | train and evaluate | fetch from OSM or OSMCha |
| **M4** (`train/setup_env.sh`) | train and evaluate | fetch from OSM or OSMCha |

Every command is resumable: stop with Ctrl+C, run it again, it continues.

---

## Step 1 — fetch (laptop, at home)

Needs `.env` with `OSMCHA_TOKEN` and `MAPGUARD_CONTACT` (see `.env.example`). Data goes to
`MAPGUARD_DATA` (default `C:\osm-data`). Keep the laptop plugged in and awake.

Progress of everything below, at any time:

```bash
python scripts/fetch_osm.py status
```

### 1a. Finish the OSMCha labels (~1–3 h, depends on osmcha.org)

The store has gaps: 2023Q4 has no harmful labels, and 2025Q4–2026Q1 have no cleared ones.

```bash
python scripts/fetch_osmcha_labels.py --since 2023-01-01 --until 2026-09-16 --no-count
```

```bash
python scripts/fetch_osmcha_labels.py --since 2023-01-01 --until 2026-09-16 --no-count --retry-gaps
```

Check `python scripts/fetch_osmcha_labels.py --stats`. If osmcha.org is returning HTTP 0/5xx
for every window, it is down — stop and try later, do not hammer it.

### 1b. Choose corpora M and N (seconds) — only after 1a is complete

```bash
python scripts/fetch_osm.py select
```

4,000 harmful + 4,000 reviewed-clean (M), water-filled across quarters, plus ~6,400 random
candidate ids for N (oversampled; 4,000 are kept). Re-running does nothing; `--force`
rebuilds, and must only be used before 1c.

### 1c. Modern metadata + osmChange contents (~3–4 h at 2 req/s)

```bash
python scripts/fetch_osm.py changesets --rate 2
```

### 1d. Previous object versions (duration unmeasured — one call per 725 objects) — can run alongside 1c

Parses every stored osmChange (Corpus A now, M/N when 1c has finished), then multi-fetches
the previous version of each modified or deleted object, 725 per call.

```bash
python scripts/fetch_osm.py versions --rate 1
```

Run it **again after 1c finishes** so M and N are planned too.

### 1e. Accounts and point-in-time histories (~2–5 h) — after 1c

```bash
python scripts/fetch_osm.py users --rate 1
```

```bash
python scripts/fetch_osm.py histories --rate 1.5
```

Keep the total OSM API rate around 4 req/s across all terminals.

### 1f. Optional — more adiff geometry, overnight

```bash
python scripts/fetch_corpus_a_geometry.py
```

~122 s per changeset, so ~300 per night. It now runs in hash order, so wherever it stops the
enriched set is an unbiased random sample. (The first 1,055 were fetched in id order and lean
towards 2014 — the build report says how many rows have adiff; B1 claims on full way
geometry must be made on that subset and must name its size.)

---

## Step 2 — build the feature store (laptop, ~20–40 min, no network)

```bash
python -m mapguard.features.build_store
```

Writes `C:\osm-data\interim\{changesets,edits,user_seq}.parquet` and `build_report.json`.

**Read `build_report.json` before going further.** For each corpus:

| Check | Acceptable | If not |
|---|---|---|
| `dropped_no_contents` | a handful | re-run 1c |
| `history_missing_label_0` vs `_label_1` | similar *rates* | re-run 1e; if still unequal, B4 results measure the fetch — report that |
| `account_age_known_label_*` | similar rates | re-run `users` |
| `edits_prev_version / edits_eligible_prev_version` | > 90% | re-run 1d |

Then copy the report into the repo so the paper's numbers have a source:

```bash
cp C:/osm-data/interim/build_report.json results/feature_store_build_report.json
```

Optional, CPU-only sanity check before anything goes to a GPU (a few minutes):

```bash
python train/01_baselines/baselines_train.py --set task=ovid
```

B0 must stay below ROC-AUC 0.85. If it does not, stop: the labels leak through metadata.

---

## Step 3 — bundle (laptop, ~2 min)

Commit first, so the bundle's git SHA describes it.

```bash
python cluster/make_notebooks.py
```

```bash
python cluster/bundle.py
```

Produces `cluster/_bundles/mapguard_bundle.zip` — code, notebooks, feature store, frozen
splits and a manifest. The same zip goes to the cluster and to the M4.

---

## Step 4a — train on the SVKM cluster

See RUN_ALTAIR.md for the portal. For every job: image `tensorflow_pbs:23.07-tf2-py3`,
1 GPU, 8 CPUs, 32 GB; upload `mapguard_bundle.zip` and the notebook for the stage.

| Order | Notebook | Time | Needs |
|---|---|---|---|
| 1 | `cluster/00_wiring_check` | < 5 min | — **run first; stop if FAIL** |
| 2 | `cluster/99_full_pipeline` with `QUICK = True` | ~15 min | 1 |
| 3 | `cluster/01_baselines` | minutes | — |
| 4 | `cluster/02_stage2_supervised` | 30–90 min | 3 |
| 5 | `cluster/04_stage1_pretrain` | 20–60 min | 3 |
| 6 | `cluster/03_stage2_ablations` | hours | 5 |
| 7 | `cluster/05_stage1_label_efficiency` | hours | 5 |
| 8 | `cluster/06_temporal_evaluation` | 30–90 min | 4 |

Jobs 6, 7 and 8 are independent of each other — submit them in parallel if slices are free.
Each notebook's last cell zips `experiments/` for download.

## Step 4b — train on the M4

```bash
unzip mapguard_bundle.zip -d mapguard && cd mapguard
```

```bash
bash train/setup_env.sh
```

```bash
source .venv-train/bin/activate && python train/run_pipeline.py --feature-store data/interim --task ovid --quick
```

```bash
python train/run_pipeline.py --feature-store data/interim --task ovid
```

Same stages, same order, same outputs as the cluster. For the modern tasks add `--task T1`
or `--task T2` (needs step 1b–1e done).

---

## What to report, and what not to

- Report runs **without** `_quick` in the directory name only.
- R1b is the result. R1a is user-leaky and exists only to compare with Ovid; label it so.
- `03` ablation deltas for B1 on the full corpus mix edits with and without full geometry;
  the build report says how many changesets have adiff.
- `04` zero-shot AUC: report it whatever it is.
- Every `results.json` has a sibling `provenance.json` with the git SHA, data fingerprints
  and split file — quote those, not memory.
