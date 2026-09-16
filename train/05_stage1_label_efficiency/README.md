# 05 · Stage 1 → Stage 2 — label efficiency (R4)

**Run order 5.** Plan stage: Stage 1 transfer into Stage 2, result **R4**. **Cluster.**
Needs a `04_stage1_pretrain` run **and** its zero-shot check.

| File | Does |
|---|---|
| `config.json` | label fractions, seeds, which encoder |
| `label_efficiency.py` | scratch vs pretrained at each label budget, then the curve |

4 budgets × 2 seeds × 2 initialisations = **16 training runs**.

## Fairness rules built in

- Same split, same validation and test fold for every point - only training rows shrink.
- **Nested budgets**: 10% ⊂ 25% ⊂ 50% ⊂ 100%, so the curve is not sampling noise.
- Class-stratified subsets - a small budget does not quietly change the class balance.
- Scratch and pretrained see **identical rows** at every (budget, seed).

The encoder was pretrained on *unlabelled* edits from the whole training fold. That is the
point of self-supervision - unlabelled data is plentiful, labels are not - and it is what
"10% of labels" means. State it in the paper.

```bash
python train/05_stage1_label_efficiency/label_efficiency.py --dry-run --set pretrained_encoder=<stage1 run dir>
python train/05_stage1_label_efficiency/label_efficiency.py --set pretrained_encoder=<stage1 run dir>
```

If the curves overlap, the finding is "label scarcity is not the bottleneck". Report it.
