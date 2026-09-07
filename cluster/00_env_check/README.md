# 00 - environment check

**Run first. Run again whenever the container image changes.**

| | |
|---|---|
| Image | `tensorflow_pbs:23.07-tf2-py3` |
| GPUs | 1 |
| Memory | 8192 MB |
| Runtime | under 2 minutes |
| Needs | nothing - no data, no credentials |
| Produces | `results/cluster_env_check.json` |

## Why it exists

The cluster image is NGC 23.07: **TensorFlow 2.13, Keras 2.x**. Gate 1d was verified
locally on Keras 3.15. The two APIs differ exactly where this model uses them, so
`src/mapguard/models/stage2.py` is written to build on both - and this notebook is the
proof, on the image that will actually train.

It also re-runs the masking check on a real model. A model that attends to padding trains
fine, scores fine, and is wrong; the failure shows up three weeks later as "geometry did
not help".

## If it fails

Fix `src/mapguard/models/stage2.py` so it builds on both versions. Do **not** pin a
different Keras inside the notebook - the model has to keep working on the laptop too.
