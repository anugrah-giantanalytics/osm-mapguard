# 04 · Stage 1 — self-supervised pretraining (conditional)

**Run order 4.** Plan stage: **Stage 1** (PROJECT_PLAN §5.2), **conditional on week 4**
(§13). **Cluster.** First thing cut if the timeline slips; nothing in 01–03 or 06 needs it.

| File | Does |
|---|---|
| `config.json` | which split to respect, encoder width, corruption list |
| `corruptions.py` | Pretext C - six synthetic damage modes, each tied to a real vandalism mode |
| `stage1_pretrain.py` | trains the edit encoder: real edit vs corrupted edit |
| `stage1_zero_shot.py` | **run before fine-tuning** - does the pretext model see real damage at all? |

## One pretext task, not three

The plan lists masked tag modelling, geometry reconstruction and corruption contrast. At
four weeks only **corruption contrast** survives, because it is the only one whose training
signal is shaped like the downstream task. The other two are future work, and the paper
should say so.

## What transfers, exactly

The encoder's first layer is identical to Stage 2's `edit_encoder`: same columns in the
same order, same width (`model.embed` must match), same normalisation statistics. Stage 2
refuses to load it if any of those differ.

To use it:

```bash
python train/02_stage2_supervised/stage2_train.py --set init.pretrained_encoder=experiments/runs/stage1_pretrain/<dir>
```

## Rules

- **No evaluation data.** Only training-fold changesets, minus any whose author appears in
  validation or test. `disjointness.json` records the overlap counts; a non-zero count aborts.
- **Pretraining validation is split by changeset**, since edits within a changeset are
  near-duplicates.
- **Per-corruption detection rates are reported**, so a mode the encoder never learned is
  visible instead of hidden inside one average.
- **Zero-shot AUC is reported either way.** Near 0.5 means the corruptions do not resemble
  real damage. That is a result about the pretext design, not a number to drop.

## Known gaps

No text corruption and no mass-deletion corruption: both signals are changeset-level in
this feature store, so a per-edit version would be corrupting columns that do not exist.

```bash
python train/04_stage1_pretrain/stage1_pretrain.py --dry-run
python train/04_stage1_pretrain/stage1_pretrain.py
python train/04_stage1_pretrain/stage1_zero_shot.py --run experiments/runs/stage1_pretrain/<dir>
```
