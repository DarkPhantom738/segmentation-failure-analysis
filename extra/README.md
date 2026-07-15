# Extra / exploratory analyses

This folder holds RQ1 probing, representation editing, and repair experiments that are **not** required for the main confidence + consistency failure-triage pipeline.

Core reproduction lives in the repository root (`README.md`).

## Prerequisites

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

You need a trained checkpoint, BraTS data or the local cache under `outputs_10hour/cache/`, and `results/paper/failure_metrics.csv` (or a local `outputs_10hour/failure_tables/failure_metrics.csv`).

## RQ1: recoverability, directions, editing

```bash
# From repo root
bash extra/scripts/run_representation_analysis.sh
```

Or step by step:

```bash
python extra/scripts/learn_semantic_directions.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --layer-index outputs_10hour/layer_embeddings/layer_embedding_index.csv \
  --failure-table results/paper/failure_metrics.csv \
  --output-dir outputs_10hour/semantic_directions

python extra/scripts/analyze_representation_editing.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table results/paper/failure_metrics.csv \
  --directions-dir outputs_10hour/semantic_directions \
  --output-dir outputs_10hour/representation_editing

python extra/scripts/analyze_layer_holdout_recoverability.py \
  --layer-index outputs_10hour/layer_embeddings/layer_embedding_index.csv \
  --failure-table results/paper/failure_metrics.csv

python extra/scripts/analyze_layer_interventions.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table results/paper/failure_metrics.csv \
  --output-dir outputs_10hour/layer_interventions
```

Library code remains under `src/analysis/` (no algorithm forks here).

## Repair feasibility (exploratory)

```bash
python extra/scripts/run_fast_spatial_edema_repair.py \
  --config extra/configs/fast_spatial_edema_repair.yaml --stage smoke

python extra/scripts/run_fast_classifier_inverted_repair.py \
  --config extra/configs/fast_classifier_inverted_repair.yaml --stage smoke

python extra/scripts/run_decoder1_repair_diagnostics.py \
  --config extra/configs/fast_spatial_edema_repair.yaml
```

Outputs write under local `outputs_*` directories (gitignored).

## Author notes

See [docs/reporting_notes.md](docs/reporting_notes.md).
