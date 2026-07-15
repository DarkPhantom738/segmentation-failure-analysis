#!/usr/bin/env bash
# End-to-end paper triage regeneration (heavy: needs data + checkpoint).
# Inspect committed numbers without re-running: results/paper/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

CONFIG="${CONFIG:-configs/ten_hour.yaml}"
CKPT="${CKPT:-outputs_10hour/checkpoints/checkpoint_latest.pt}"
FAILURE="${FAILURE:-results/paper/failure_metrics.csv}"
if [[ ! -f "$FAILURE" ]]; then
  FAILURE=outputs_10hour/failure_tables/failure_metrics.csv
fi

echo "=== 1) TTA export (optional if metrics already exist) ==="
if [[ "${SKIP_TTA:-0}" != "1" ]]; then
  python train.py --config "$CONFIG" --export-tta --checkpoint "$CKPT"
  python analyze_failures.py \
    --metrics outputs_10hour/metrics_uncertainty.csv \
    --output outputs_10hour/failure_tables/failure_metrics.csv
  FAILURE=outputs_10hour/failure_tables/failure_metrics.csv
fi

echo "=== 2) Layer embeddings ==="
python export_layer_embeddings.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --failure-table "$FAILURE" \
  --output-dir outputs_10hour/layer_embeddings

echo "=== 3) Confidence features ==="
python scripts/run_layer_aware_latent_risk.py \
  --config configs/layer_aware_latent_risk.yaml \
  --stage confidence

echo "=== 4) Consistency ==="
python scripts/run_consistency_failure_detection.py \
  --config configs/consistency_failure_detection.yaml

echo "=== 5) Triage ==="
python scripts/run_confidence_consistency_triage.py \
  --config configs/confidence_consistency_triage.yaml

echo "=== 6) Method validation ==="
python scripts/run_method_validation.py \
  --config configs/method_validation.yaml

echo "=== DONE ==="
echo "New run artifacts are under local outputs_* (gitignored)."
echo "Canonical committed snapshot: results/paper/"
