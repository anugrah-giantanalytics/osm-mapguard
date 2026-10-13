# MapGuard on Altair (SVKM cluster)

One zip per training round. The job script unpacks it into its own round folder, runs every
training step on the GPU, then the WHERE data job on the CPU, and writes logs, timings and one
results zip for the round.

## Build (laptop)

```bash
python altair/build_zip.py --round r3_v3_explore --plan altair/plans/r3_v3_explore.txt
```

Writes `altair/_dist/mapguard_altair.zip` (git-ignored: it carries data, and
`where_targets.parquet` holds raw OSM user ids for the WHERE job).

## Run (Jupyter terminal on the cluster)

Upload the zip to `/data/mpstme-tanay/deepLearning_osm/`, then:

```bash
cd /data/mpstme-tanay/deepLearning_osm && unzip -o -j mapguard_altair.zip altair/run_round2.sh -d . && MAPGUARD_BASE="$(pwd)" setsid nohup bash run_round2.sh > launch.out 2>&1 < /dev/null &
```

The first command takes the job script out of the zip, so only the zip needs uploading.
Close the browser tab afterwards; do not press Quit and do not stop the Jupyter job.

## Folder layout on the cluster

```
deepLearning_osm/
  mapguard_altair.zip, run_round2.sh, launch.out
  rounds/<round>/                 one per zip; the same zip again = resume
    app/                          code + data + experiments/runs
    logs/job.log  STATUS  timing.csv  env.txt  steps/<step>.log
    <round>_results.zip           send this back
  where/                          WHERE job, shared by all rounds (done once)
    changesets-latest.osm.bz2     the planet changeset dump (~9 GB)
    author_points.npz             target authors' changeset locations
    where_features.parquet        one row per changeset id - also inside the results zip
    where.log  STATUS
```

## Check progress

```bash
cd /data/mpstme-tanay/deepLearning_osm && cat rounds/*/logs/STATUS where/STATUS 2>/dev/null
```

## Steps = the round's plan

Each zip carries `altair/PLAN.txt` (copied from `altair/plans/<round>.txt`), one step per line:

```
name | flags | run_pipeline.py arguments
```

flags: `fatal` (stop the job if it fails), `install_st` (install torch + sentence-transformers
into `.pydeps_st2`, visible to that step only), `where` (run the WHERE job). Re-running the
same zip skips finished steps. A round can only run once at a time: a second launch
exits with "already running" (lock with PID + heartbeat in `rounds/<round>/.lock`). Plans so far:

| Plan | What | Time (H100, 1 CPU core) |
|---|---|---|
| `r3_v3_explore.txt` | embeddings, smoke, v3 ablations (5 rows × 3 seeds), all six tasks with hybrid variants + stacking | ~2 h (ran 2026-10-12) |
| `r4_seeds.txt` | embeddings (fixed env), smoke, ablations 4 rows × 3 seeds, all six tasks × 3 seeds, temporal | ~4.5–5 h |

Training results are packed into the results zip **before** WHERE starts, and again at the end.
