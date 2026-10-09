# MapGuard dataset — v1.1.0

Generated 2026-09-21T19:28:46+00:00 from commit `58cb92d3f0d2`.

Training-ready: `feature_store/` is the exact input of `train/` (pseudonymised), with the
frozen splits in `splits/`. Build provenance and per-label coverage are in
`feature_store/build_report.json`; every column is documented in
`feature_store/feature_schema.json`.

| Corpus | Years | Label |
|---|---|---|
| A | 2014–2019 | reverted (Ovid benchmark) |
| R | 2021-09 → 2026-06 | reverted by another account within 90 days, mined from the planet changeset dump; `revert_mentions_vandalism` marks Ovid's stricter rule |
| M | 2023 → 2026 | OSMCha human review (harmful vs flagged-and-cleared) |
| N | 2023 → 2026 | random unreviewed, presumed clean |

## What this is

Changesets from OpenStreetMap labelled as damaging or not, for training and evaluating
vandalism-detection models. Two label sources with **different meanings** — see *Labels*.

## Contents

| File | Rows | Description |
|---|---|---|
| `feature_store/*.parquet` | 46,578 changesets | training input for `train/` - changesets, sampled edits, point-in-time user history |
| `corpus_r_reverts.parquet` | 16,562 | which revert labelled each corpus R positive |
| `splits/` | — | frozen train/val/test id lists |
| `corpus_a_changesets.parquet` | 18,276 | Ovid-benchmark changesets, 2014–2019, with parsed edit counts |
| `osmcha_labels.parquet` | 133,523 | OSMCha human review decisions, 2023 onward |
| `fetch_coverage.json` | 1687 | windows the harvest failed to retrieve |
| `results/` | — | gate artefacts: every measurement behind the design decisions |
| `fixtures/` | — | real changesets selected as awkward cases |
| `MANIFEST.json` | — | provenance, counts, git SHA |

`corpus_a_changesets.parquet` covers **18,276** changesets, of which
**18,276** had their full edit contents fetched and parsed, totalling
**2,065,047 edits**.

## Labels — read this before using them

| Source | Meaning | Caveat |
|---|---|---|
| `ovid_revert_derived` | the changeset was later reverted | A revert is **not** a synonym for vandalism. Imports, accidents and mapping disputes get reverted too. Class balance is a constructed 50/50, not natural prevalence. |
| `osmcha_human_review` | a person reviewed it on osmcha.org | Reviewers only open changesets an automatic filter already flagged, so `harmful=false` here means **"flagged and cleared"**, not "clean". Both classes come from the suspicious tail. |

Real-world vandalism prevalence is well under 1%. Any metric computed on the balanced
set overstates operational performance substantially.

## Geography — Corpus A

| Region | Changesets | Share |
|---|---|---|
| Europe | 7,845 | 42.9% |
| North America | 3,724 | 20.4% |
| Asia | 1,973 | 10.8% |
| Africa | 1,762 | 9.6% |
| other | 1,188 | 6.5% |
| South America | 1,032 | 5.7% |
| Oceania | 523 | 2.9% |
| unknown | 229 | 1.2% |

Europe-heavy, reflecting OpenStreetMap's real contributor distribution rather than a
sampling choice. Models evaluated on it are best evidenced on European edits.

## Coverage and gaps

`fetch_coverage.json` lists every date window the harvest failed to retrieve, with
status and retry count. Unresolved gaps at packaging time: **353**.
A dataset that cannot say what it failed to collect cannot support a coverage claim, so
the gaps ship with the data rather than being quietly dropped.

## Privacy

OpenStreetMap user ids are replaced with a **salted SHA-256 digest** (`user_pseudonym`,
16 hex chars). Usernames and OSMCha reviewer names are **removed entirely**. The salt is
not published.

Changeset ids are retained because they are already public and the data is
reconstructible from them through the OSM API — but do not attempt to re-identify
contributors, and do not use this dataset to make claims about individual accounts. A
label here is a judgement about an edit, not about a person.

## Licence

**ODbL-1.0. © OpenStreetMap contributors.**

This is a Derivative Database of OpenStreetMap data and must stay under ODbL. See
`LICENSE-DATA` in the archive. The repository's source code is licensed separately.

## Reproducing

Every changeset id is in the archive. Contents are re-fetchable from the OSM API:

    GET https://api.openstreetmap.org/api/0.6/changeset/<id>/download

Gate artefacts under `results/` record the measurements behind every design decision —
API latency, Overpass throughput, label coverage by year, and the leakage tripwire.
