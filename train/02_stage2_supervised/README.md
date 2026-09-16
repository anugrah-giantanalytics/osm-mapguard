# 02 · Stage 2 — supervised classifier

**Run order 2.** Plan stage: **Stage 2** (PROJECT_PLAN §5.3). Results: R1a, R1b, R2.

The five-branch model that the paper is about. Everything after this folder either varies
it (03 ablations, 05 label efficiency) or moves it in time (06).

| File | Does |
|---|---|
| `config.json` | task, split, model and training settings |
| `stage2_train.py` | trains one model, evaluates it, writes a run directory |
| `stage2_evaluate.py` | re-scores a finished run, including on another task |

## Before running

1. `00_wiring_check` passed on this machine.
2. `01_baselines` done for the same task and split — it gives the number to beat, and it
   creates the frozen split this stage reuses.
3. The feature store satisfies `train/common/data_contract.py`.

```bash
python train/02_stage2_supervised/stage2_train.py --dry-run
```

validates config, data and split without training.

## The runs the paper needs

```bash
python train/02_stage2_supervised/stage2_train.py --set task=T2 --set split=R1b
python train/02_stage2_supervised/stage2_train.py --set task=T1 --set split=R1b
python train/02_stage2_supervised/stage2_train.py --set task=ovid --set split=R1a --set tag=ovid_protocol
python train/02_stage2_supervised/stage2_train.py --set task=ovid --set split=R1b
```

Then the T1/T2 gap:

```bash
python train/02_stage2_supervised/stage2_evaluate.py --run <T2 R1b run dir> --task T1
```

## What a run directory holds

| File | |
|---|---|
| `config.json` | resolved config, overrides applied |
| `provenance.json` | git sha, library versions, device, data and split fingerprints |
| `model_config.json` | exact build arguments and feature column order |
| `normaliser.json` | training-row statistics - evaluation must use these |
| `best.weights.h5`, `model.weights.h5` | best-on-validation and final (restored best) weights |
| `training_log.csv` | per-epoch loss and validation PR-AUC |
| `predictions.npz` | val and test scores with cids, for plots and later analysis |
| `results.json` | metrics with 95% intervals, calibration, attribution, masking checks |

## Decisions baked into the training loop

- **Early stopping on validation PR-AUC**, never accuracy.
- **Threshold chosen on validation**, applied to test.
- **Masking verified on the real model before and after training.** A failure aborts.
- **Focal loss by default**; `--set train.loss=bce` for the comparison row in 03.
- **Weights saved, not models** - custom layers serialise differently across Keras 2 and 3.

## Timing

On an M4, time one epoch with `--set device=cpu --set train.epochs=1` and again with
`device=auto`. Masked LSTM/GRU layers often run slower on Metal than on the CPU cores.
