# 00 · Wiring check — run first, produces no results

**Run order 0.** Not a plan stage. Run it on every new machine, container image or
TensorFlow version, before anything in 01–06.

| File | Does |
|---|---|
| `wiring_check.py` | pushes random tensors through every real code path, reports PASS/FAIL |

```bash
python train/00_wiring_check/wiring_check.py
python train/00_wiring_check/wiring_check.py --device cpu
```

**Random tensors, not data.** Nothing is written to `experiments/`, `results/` or the
split directory; everything happens in a temp directory that is deleted afterwards.

It checks:

- every ablation branch set builds, trains one epoch, and **masking holds** before and after
- changesets with **zero edits** and users with **no history** do not produce NaN
- weights **round-trip** through `model_config.json` + rebuild + load
- Stage 1 corruptions run, the encoder trains, and its weights **transfer** into Stage 2
- an encoder pretrained on a **different split is refused**

The last one matters: an encoder pretrained on another split may have seen this run's test
rows, and the refusal is what stops that becoming a "pretrained beats scratch" result.

Expected on the SVKM cluster image (TF 2.13, Keras 2) and on a current TF (Keras 3): PASS on
both. A FAIL on one and not the other is a Keras-version bug in `src/mapguard/models/stage2.py`.
