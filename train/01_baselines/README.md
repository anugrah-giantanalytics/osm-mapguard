# 01 · Baselines

**Run order 1.** Plan stage: baselines (PROJECT_PLAN §11). Results: R1a, R1b, R2 reference
numbers. **CPU only** — scikit-learn and XGBoost, no TensorFlow.

| File | Does |
|---|---|
| `config.json` | task, split, which baselines to run |
| `baselines_train.py` | B0 tripwire, logistic regression, XGBoost → one run directory |

| Model | Sees | Why it exists |
|---|---|---|
| **B0** | comment length, editor string, account age | the leakage tripwire - nothing about what was edited |
| **LogReg** | all tabular features | the floor a neural model must clear |
| **XGBoost** | all tabular features | the strongest non-neural reference |

**Read B0 first.** Its verdict uses ROC-AUC, not F1: on a balanced set, predicting
"damaging" for everything already scores F1 0.667, so an F1 threshold catches nothing.
AUC ≥ 0.85 from metadata alone means the labels leak and nothing downstream is trustworthy.

This stage also **creates the frozen split** in `experiments/splits/`. Run it before Stage 2
so both train on exactly the same rows.

```bash
python train/01_baselines/baselines_train.py --set task=T2 --set split=R1b
python train/01_baselines/baselines_train.py --set task=T1 --set split=R1b
python train/01_baselines/baselines_train.py --set task=ovid --set split=R1a --set tag=ovid_protocol
python train/01_baselines/baselines_train.py --set task=ovid --set split=R1b
```

If Stage 2 cannot beat XGBoost on the same split, that goes in the paper as a finding.
