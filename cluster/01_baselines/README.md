# 01 - baselines

| | |
|---|---|
| Image | `tensorflow_pbs:23.07-tf2-py3` |
| GPUs | 1 (CPU-bound in practice) |
| Memory | 16384 MB |
| Runtime | ~10 minutes |
| Needs | `changesets.parquet`, and `00_env_check` passed |
| Produces | `results/baselines.json` |

The number Stage 2 has to beat. B0 tripwire, logistic regression, XGBoost, under both
splits with bootstrap confidence intervals.

**R1a minus R1b is a finding, not a nuisance.** It measures how much of a published score
is account memorisation rather than edit understanding.
