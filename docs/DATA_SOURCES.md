# Data sources — what each one can and cannot give this project

MapGuard classifies **changesets** (one editing session) as damaging or not. A training row
therefore needs three things, and a source is only useful if it supplies at least one:

1. **a label** — someone judged this changeset damaging, or it was reverted;
2. **the edit itself** — which objects changed, and how (osmChange);
3. **the state before the edit** — to measure displacement, shape change and tag diffs.

Everything below was checked against the live services on 2026-09-17/18.

## Sources that were proposed

| Source | What it serves | Label | Edit | Before-state | Used? |
|---|---|---|---|---|---|
| [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) | current or historical *map state* for a query; `[adiff:t1,t2]` gives before/after geometry for a bbox and time window | no | partly | **yes, with full way geometry** | yes: geometry route 2 (`scripts/fetch_corpus_a_geometry.py`) |
| [Geofabrik](https://download.geofabrik.de/) | regional extracts of the **current** map (`.osm.pbf`, shapefiles); daily `.osc` diffs | no | no (diffs are the same as planet replication) | no | no, see below |
| [QGIS QuickOSM](https://plugins.qgis.org/plugins/QuickOSM/) | a QGIS GUI over Overpass for current features | no | no | no | no: not a pipeline tool; at most for demo maps |
| [OSMnx](https://osmnx.readthedocs.io/) | street networks and geometries of the current map, as graphs and dataframes | no | no | no | no |
| [Changesets API](https://wiki.openstreetmap.org/wiki/API_v0.6#Retrieving_changesets) | changeset metadata, `/download` (osmChange), per-user and per-id queries | no | **yes** | previous versions via `/nodes?nodes=idvN` | yes: everything in `scripts/fetch_osm.py` |
| [OSM Latest Changes](https://wiki.openstreetmap.org/wiki/OSM_Latest_Changes) | a web page for eyeballing recent edits in an area | no | view only | view only | no: manual; useful for spot-checking labels |
| [JOSM](https://josm.openstreetmap.de/) | desktop editor; its **reverter plugin** writes `created_by=reverter_plugin` and "Revert changeset N" | indirectly | manual | manual | indirectly: reverts it makes are mined as labels |

Why most of these do not fit:

- **Geofabrik, OSMnx and QuickOSM describe the map as it is, not how it changed.** A
  current-state extract cannot say who edited what, when, or what was there before, so it
  cannot produce a single training row. Geofabrik's *full-history* extracts (`.osh.pbf`,
  behind an OSM login) do contain every version, but only per region. The labelled changesets
  here are worldwide (OSMCha reviews and reverts come from everywhere), so a regional file
  covers a small fraction of them. The planet full-history file is well over 100 GB. It is the
  right tool if the project is ever narrowed to one country; it is not needed for this one.
- **Overpass is kept, but only for what nothing else gives: full before/after way geometry.**
  It measured ~122 s per changeset on the public instance (two slots per client), so it
  covers a subset, fetched in hash order so that any stopping point is an unbiased sample.
- **The Changesets API is the backbone, but it cannot search.** It returns a changeset by
  id, a user's changesets, or 100 ids at once. It cannot find "changesets that were reverted".

## Sources this project adds

| Source | Gives | Why |
|---|---|---|
| **Planet changeset dump** `changesets-latest.osm.bz2` (8.76 GB, file `changesets-260914`) | metadata and tags of every changeset since 2005 | The only bulk source of **revert** information: tool tags (`revert:id`) and comments ("Reverting 184632895 by …"). It is multistream bzip2 sorted by id, so only the 2021-09 onward byte range (~3.9 GB) is read, in parallel ranges, without storing the file (`scripts/mine_changesets.py`). |
| **OSMCha API** | human review: harmful / reviewed-and-cleared | The only human-judged labels. **Measured: zero reviewed changesets in 2018–2022**; reviews start 2023-01 (`results/gate2a_osmcha_coverage.json`). |
| **Ovid benchmark labels** (`data/labels/ovid_labels.tsv`) | 18,276 revert-derived labels, 2014–2019 | The published baseline to compare with. |
| **OSM API multi-fetch** `/nodes?nodes=idvN,...` (725 per call) | the previous version of every modified or deleted object | Exact tag diffs, node displacement and way node-count change for every changeset, at ~1 call per 725 objects instead of ~122 s per changeset. |

## What "4–5 years of recent data" can honestly mean

| Corpus | Years | Label | Size target | Source |
|---|---|---|---|---|
| **A** | 2014–2019 | reverted (Ovid) | 18,276 | Ovid + API |
| **R** | **2021-09 → 2026-06 (4.8 years)** | reverted by another account within 90 days (Ovid's definition, recomputed) | 6,000 + 6,000 | changeset dump + API |
| **M** | 2023-01 → 2026-09 (3.7 years) | human-reviewed harmful vs cleared | 4,000 + 4,000 | OSMCha + API |
| **N** | 2023-01 → 2026-09 | random unreviewed (presumed clean) | 4,000 | API |

Human-reviewed labels **cannot** reach back 4–5 years: OSMCha has none before 2023. Revert
labels can, and they use the same definition as the benchmark, which is what makes the
temporal claim testable. Train on A (2014–2019), test on R (2021–2026), and any drop is
time, not a change of labeller (`--task ovid_modern`).

The window stops at 2026-06-16, 90 days before the dump date. A changeset from last week has
not had time to be reverted, and counting it as a negative would be wrong.

### Known weaknesses of revert labels (report them)

- A revert is not always vandalism: it can mean a disagreement, an undiscussed import or a
  good-faith mistake. The Ovid benchmark has the same property, and M exists to measure it
  with human judgement.
- Mass reverts of one account (e.g. `simple_revert.py` runs) could dominate the positives,
  so each author contributes at most 5 changesets per class.
- Old changesets reverted years later (cleanup bots) are not "caught in review", so revert
  delay is capped at 90 days.
- Self-reverts (an author undoing their own edit) are excluded, and so are partial reverts
  and reworks.
- Negatives are "not reverted by the dump date", not "verified clean". Some vandalism is
  never reverted.
