# Contributing

Four people, one repository, a laptop with 0.77 GB of free RAM, and a public API that
belongs to volunteers. These rules exist because of those four facts.

## Branches and commits

- Branch names: `ws1/…`, `ws2/…`, `ws3/…`, `ws4/…` for workstream work, `fix/…` otherwise.
  No tool-named branches.
- [Conventional Commits](https://www.conventionalcommits.org): `feat(data): …`,
  `fix(features): …`, `docs: …`, `test: …`, `chore: …`.
- No `Co-Authored-By` trailers for tooling, no generated-by footers.
- One logical change per commit. A commit that both moves a file and changes its
  behaviour is unreviewable.

## What never goes in the repository

- **Bulk data.** It lives under `MAPGUARD_DATA` (default `C:\osm-data`), outside the repo
  and outside OneDrive. `data/raw`, `data/interim`, `data/processed` and `*.parquet` are
  gitignored; the label TSV is the one tracked exception because it defines the benchmark.
- **Secrets.** The OSMCha token goes in a gitignored `.env`. Copy `.env.example` to start.
- **Model weights or checkpoints.** They go to Drive.
- **Screenshots of numbers.** Metrics go to `results/*.json` so the paper can cite a file.

## Talking to OSM

The API and Overpass are donated infrastructure and this project is a guest on them.

- Go through `mapguard.data.http.PoliteSession`. It holds a *global* token bucket; a
  per-thread sleep is not a rate limit.
- Real User-Agent with a contact address. `mapguard.config.USER_AGENT` builds it.
- HTTP 509 and 429 mean slow down. The session backs off for every thread, not just the
  one that got the response. Do not work around it.
- Cache to disk and resume. A rerun should cost nothing.
- Overpass runs two slots per client. Do not run two Overpass jobs at once, and do not
  send it a planet-sized bbox - `should_use_adiff` is the guard, use it.

## Memory discipline

The development laptop has under 1 GB free. Anything that loads a whole corpus into a
dataframe will thrash it.

- Stream XML with `iterparse`, and detach elements, do not just clear them.
- Query Parquet with DuckDB rather than reading it into pandas.
- Heavy work belongs on Colab (12.7 GB) or Kaggle (~29 GB), not locally.

## Correctness rules that are not style

Three mistakes would invalidate results rather than merely look bad. They have tests;
keep them passing.

1. **User features are point-in-time.** Only changesets strictly before the target's
   `created_at`. Anything else lets the model see that the account was blocked next year.
   (`account_created` is the one exception - it is fixed for the life of the account.)
2. **Splits are user-disjoint** when reporting the honest number. The leaky random split
   is reported too, as R1a, and always labelled as leaky.
3. **Pretraining excludes the evaluation set** - by changeset id *and* by uid.

## Notebooks

Thin. `git pull`, `pip install -r requirements.txt`, import from `src/`, run. No
2000-line cells: a notebook that holds logic cannot be reviewed or diffed.

## Before you push

```bash
python -m pytest -q
```

If you touched a gate, rerun it and commit the JSON. A gate with no file in `results/`
did not run.
