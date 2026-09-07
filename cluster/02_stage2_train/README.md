# 02 - Stage 2 training

| | |
|---|---|
| Image | `tensorflow_pbs:23.07-tf2-py3` |
| GPUs | 1 |
| Processors | 8 |
| Memory | 32768 MB |
| Runtime | 1-3 hours depending on branches |
| Needs | all three Parquet tables, `00` and `01` done |
| Produces | `results/stage2_r1b.json`, `experiments/checkpoints/stage2_final.keras` |

The research result. Five branches, masked attention fusion, early stopping on validation
**PR-AUC** (never accuracy).

Checkpoints every epoch. If the job dies, resume from the last
`experiments/checkpoints/stage2_epoch*.weights.h5`.

Download both the results JSON and the checkpoint before terminating the job.
