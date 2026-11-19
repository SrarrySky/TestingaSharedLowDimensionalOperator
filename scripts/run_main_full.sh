#!/usr/bin/env bash
set -euo pipefail

# Ubuntu launcher for the complete publication pipeline.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
config="${1:-$root/config/main_runner.json}"
output="${2:-$root/outputs/main_run}"
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))

cd "$root"

# Train-only causal support.
"$python_bin" -m cli causal \
  --data data/processed/dataset.npz \
  --config config/causal.yaml \
  --output outputs/checkpoints/causal_support.npz \
  --jobs 3

# Fit DLC-SSM on the Train split.
"$python_bin" -m cli fit \
  --data data/processed/dataset.npz \
  --model-config config/dlcssm.yaml \
  --support outputs/checkpoints/causal_support.npz \
  --output outputs/checkpoints/dlcssm_train.npz

# Synthetic smoke experiment.
"$python_bin" -m cli synthetic --scenario shared_low_rank --output outputs/checkpoints/synthetic_demo.npz

exec "$python_bin" -B "$root/scripts/run_main.py" \
  --config "$config" \
  --output "$output" \
  --threads 8 \
  "$@"
