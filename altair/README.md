# MapGuard on Altair (SVKM cluster) — round 2

One zip, one job script. The job script finds the folder, unpacks code + data, installs what
the image lacks, runs every round-2 step, and writes logs, timings and a results zip.

## Files

| File | What |
|---|---|
| `_dist/mapguard_altair.zip` | code + pseudonymised feature store + these scripts (built by `build_zip.py`) |
| `_dist/run_round2.sh` | **Job Script**: the full round 2, resumable |
| `_dist/smoke_test.sh` | optional Job Script: only the ~10 min smoke test |

## Before the first job

Put `mapguard_altair.zip` in `/data/mpstme-tanay/deepLearning_osm/` (or upload it as Job Files;
both work).

## The form

| Field | Value |
|---|---|
| Job Name | `mapguard_r2` |
| Container Image | `tensorflow_pbs:23.07-tf2-py3` |
| Job Script | `run_round2.sh` |
| Job Files | `mapguard_altair.zip` |
| Output Directory | `/data/mpstme-tanay/deepLearning_osm/altair_output` |
| Number of GPUs per Node | `1` (default) |
| Processors per Node / Memory (MB) | `8` / `32768` if you can change them (the data needs a few GB) |

## What it runs

| Step | Task / stages | GPU time (estimate) |
|---|---|---|
| smoke | ovid, quick, all stages — stops the job if it fails | ~10 min |
| main | ovid 00,01,02,07 | ~15–30 min |
| ablations | ovid 03 (8 rows × 2 seeds) | ~1–2 h |
| modern | ovid_modern 01,02,07,06 | ~20–30 min |
| T1, T2 | 01,02,07,06 | ~20 min each |
| revert_vandalism, revert | 01,02,07 | ~15 min each |

## Where things land

`/data/mpstme-tanay/deepLearning_osm/outputs/<job>_<id>_<time>/`

- `job.log` — everything, live
- `STATUS` — current step, or how it ended
- `timing.csv` — start, end, seconds and exit code per step
- `env.txt` — GPU, TensorFlow, packages
- `steps/<step>.log` — one log per step
- `round2_results.zip` — all runs (metrics, predictions, `best.weights.h5` checkpoints,
  per-epoch `training_log.csv`) + logs. **Send this back.**

Runs accumulate in `deepLearning_osm/mapguard_v2/experiments/runs/` across jobs.

## If it stops

Submit the same job again. Finished steps are skipped (markers in `mapguard_v2/.done/`); it
resumes at the first unfinished one. A new zip re-runs everything. Run one job at a time —
two jobs would share the same `mapguard_v2/` folder.

If the Jupyter profile starts a notebook instead of running the script: open a Terminal in
Jupyter and run `bash /data/mpstme-tanay/deepLearning_osm/mapguard_v2/altair/run_round2.sh`
(after the zip has been unpacked once), or unzip it there first.
