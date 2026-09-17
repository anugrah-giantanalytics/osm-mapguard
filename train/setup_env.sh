#!/usr/bin/env bash
# Create the training environment on an M4 Mac or a Linux box, then prove it works.
#
#   bash train/setup_env.sh            # CPU TensorFlow (safe default on the M4)
#   bash train/setup_env.sh --metal    # also try Apple's GPU plugin
#
# Then:
#   source .venv-train/bin/activate
#   python train/run_pipeline.py --feature-store data/interim --task ovid --quick
#
# The cluster does not need this: its tensorflow_pbs image already has TensorFlow, and the
# notebooks install whatever else is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

METAL=0
[[ "${1:-}" == "--metal" ]] && METAL=1

PY=""
for cand in python3.12 python3.11 python3.10; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "Need Python 3.10-3.12 (TensorFlow has no wheels for 3.13+)."
  echo "On a Mac: brew install python@3.12"
  exit 1
fi
echo "using $($PY --version)"

$PY -m venv .venv-train
# shellcheck disable=SC1091
source .venv-train/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-train.txt

if [[ "$(uname -s)" == "Darwin" ]]; then
  python -m pip install -q "tensorflow>=2.16,<2.20"
  if [[ $METAL == 1 ]]; then
    # Optional. If the model misbehaves on Metal, uninstall it: CPU on an M4 is fast
    # enough for this model, and correctness matters more than minutes.
    python -m pip install -q tensorflow-metal || echo "tensorflow-metal unavailable; staying on CPU"
  fi
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    python -m pip install -q "tensorflow[and-cuda]>=2.16,<2.20"
  else
    python -m pip install -q "tensorflow>=2.16,<2.20"
  fi
fi

python - <<'PY'
import sys, tensorflow as tf
print("python", sys.version.split()[0], "tensorflow", tf.__version__,
      "devices", [d.name for d in tf.config.list_physical_devices()])
PY

echo
echo "Running the wiring check (random tensors, no results, a few minutes)..."
python train/00_wiring_check/wiring_check.py
