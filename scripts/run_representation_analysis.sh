#!/usr/bin/env bash
# Probing → semantic directions → representation editing → probe screens.
# Requires: trained checkpoint, layer embeddings, failure_metrics.csv
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

CONFIG=configs/ten_hour.yaml
CKPT=outputs_10hour/checkpoints/checkpoint_latest.pt
FAILURE=outputs_10hour/failure_tables/failure_metrics.csv
LAYER_INDEX=outputs_10hour/layer_embeddings/layer_index.csv

echo "=== Learn semantic directions ==="
python learn_semantic_directions.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --layer-index "$LAYER_INDEX" \
  --failure-table "$FAILURE" \
  --output-dir outputs_10hour/semantic_directions

echo "=== Representation editing (375 cases; long-running) ==="
python analyze_representation_editing.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --failure-table "$FAILURE" \
  --directions-dir outputs_10hour/semantic_directions \
  --output-dir outputs_10hour/representation_editing

echo "=== Edema probe screen (decoder1, 30 cases) ==="
python analyze_edema_probe_screen.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --failure-table "$FAILURE"

echo "=== Volume probe screen (decoder2, 30 cases) ==="
python analyze_volume_probe_screen.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --failure-table "$FAILURE" \
  --n-random 5

echo "=== Done ==="
echo "Results: outputs_10hour/representation_editing/, edema_probe_screen/, volume_probe_screen/"
