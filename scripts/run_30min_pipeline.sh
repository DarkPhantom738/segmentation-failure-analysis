#!/usr/bin/env bash
# ~30 minute Failure Cartography pipeline (fixed preprocessing/baselines).
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p outputs_30min

CONFIG=configs/thirty_min.yaml
export LOKY_MAX_CPU_COUNT=1
export MPLCONFIGDIR=/tmp/mpl

echo "=== Training (~30 min budget) ==="
python train.py --config "$CONFIG"

echo "=== TTA uncertainty export ==="
python train.py --config "$CONFIG" --export-uncertainty \
  --checkpoint outputs_30min/checkpoints/checkpoint_latest.pt

echo "=== Failure analysis ==="
python analyze_failures.py \
  --metrics outputs_30min/metrics_uncertainty.csv \
  --output outputs_30min/failure_tables/failure_metrics.csv

echo "=== Geometry analysis ==="
python analyze_geometry.py \
  --failure-table outputs_30min/failure_tables/failure_metrics.csv \
  --output-dir outputs_30min/geometry

echo "=== Baseline comparison ==="
python compare_baselines.py \
  --failure-table outputs_30min/failure_tables/failure_metrics.csv \
  --geometry-table outputs_30min/geometry/umap_coordinates.csv \
  --output-dir outputs_30min/baselines

echo "=== Done ==="
