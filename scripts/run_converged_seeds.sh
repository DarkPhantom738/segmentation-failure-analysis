#!/usr/bin/env bash
# Sequential five-seed converged U-Net training.
# Does not touch outputs_10hour/ (cache is read-only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
mkdir -p outputs_converged/logs

SEEDS=(42 123 2026 31415 271828)
for seed in "${SEEDS[@]}"; do
  if [[ "$seed" -lt 1000 ]]; then
    tag=$(printf "seed_%03d" "$seed")
  else
    tag="seed_${seed}"
  fi
  summary="outputs_converged/${tag}/convergence_summary.json"
  if [[ -f "$summary" ]]; then
    echo "[skip] $tag already has convergence_summary.json"
    continue
  fi
  echo "[train] starting model seed=${seed} -> ${tag}"
  python -u train_converged.py \
    --config configs/converged_unet.yaml \
    --seed "$seed" \
    2>&1 | tee "outputs_converged/logs/train_${tag}.log"
  echo "[train] finished seed=${seed}"
done
echo "All requested seeds complete."
