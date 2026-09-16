# 03 · Stage 2 — ablation grid

**Run order 3.** Plan stage: **Stage 2 ablations** (PROJECT_PLAN §11, cut to 7 rows × 2 seeds
by §13). **Cluster** — 14 training runs.

| File | Does |
|---|---|
| `config.json` | shared settings, seeds, which rows to run |
| `rows/*.json` | one file per table row: its name, purpose and config overrides |
| `run_ablations.py` | trains every row × seed, then writes the table |

## The rows

| Row | Branches | Answers |
|---|---|---|
| `base_edits_tabular` | B5 (+ edit trunk, no geometry) | the common base every other row is compared to |
| `plus_geometry` | B1 B5 | does edit geometry help? (G1) |
| `plus_text` | B2 B5 | does reading tag text help? (G2) |
| `plus_sequences` | B3 B4 B5 | do edit order and user history help? (G3, G4) |
| `full_scratch` | all | the Stage 2 headline |
| `full_scratch_bce` | all, BCE loss | is focal loss earning its place? |
| `full_pretrained` | all, Stage 1 encoder | does pretraining help? - **skipped unless 04 ran** |

**The edit trunk is in every row.** Attention over edits is how the model fuses anything, so
"B1 off" removes the geometry *columns* from edits and tabular features - it does not remove
edits. That is also why the base row is named for what it contains.

## Rules

- **Seeds change initialisation, not the split.** Every row trains on the same frozen split.
  Otherwise a row can look better by drawing an easier test fold.
- **Every row uses the same training loop** as stage 02. Rows differ only by their row file.
- **Attribution is off** here to save time; run `stage2_evaluate.py` on the rows you report.
- Two seeds give a mean and a spread, not a significance test. Say so in the paper.

```bash
python train/03_stage2_ablations/run_ablations.py --dry-run
python train/03_stage2_ablations/run_ablations.py
```

To add a row, add a file to `rows/`. Nothing else changes.
