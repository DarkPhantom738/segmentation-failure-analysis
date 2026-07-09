#!/usr/bin/env bash
# Mid-scale Failure Cartography pipeline: 300 cases, 50 epochs, 128^3, 1mm.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p outputs_midscale

CONFIG=configs/midscale.yaml
export LOKY_MAX_CPU_COUNT=1
export MPLCONFIGDIR=/tmp/mpl

echo "=== Training (mid-scale) ==="
python train.py --config "$CONFIG"

echo "=== TTA uncertainty export ==="
python train.py --config "$CONFIG" --export-uncertainty \
  --checkpoint outputs_midscale/checkpoints/checkpoint_latest.pt

echo "=== Failure analysis ==="
python analyze_failures.py \
  --metrics outputs_midscale/metrics_uncertainty.csv \
  --output outputs_midscale/failure_tables/failure_metrics.csv

echo "=== Geometry analysis ==="
python analyze_geometry.py \
  --failure-table outputs_midscale/failure_tables/failure_metrics.csv \
  --output-dir outputs_midscale/geometry

echo "=== Baseline comparison ==="
python compare_baselines.py \
  --failure-table outputs_midscale/failure_tables/failure_metrics.csv \
  --geometry-table outputs_midscale/geometry/umap_coordinates.csv \
  --output-dir outputs_midscale/baselines

echo "=== Done ==="
echo "Check:"
echo "  Does uncertainty still beat embeddings?"
echo "  Do embeddings cluster failure types?"
echo "  Does combined > uncertainty-only?"
