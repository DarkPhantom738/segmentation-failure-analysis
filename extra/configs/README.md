# Extra configs (`extra/configs/`)

YAML for exploratory repair experiments. These are **not** the paper triage configs (those live in [`../../configs/`](../../configs/README.md)).

## Files

| File | Used by | Purpose |
|---|---|---|
| `fast_spatial_edema_repair.yaml` | `extra/scripts/run_fast_spatial_edema_repair.py`, `run_decoder1_repair_diagnostics.py` | Paths, stages, and hyperparameters for the spatially gated decoder1 edema repair feasibility runs. |
| `fast_classifier_inverted_repair.yaml` | `extra/scripts/run_fast_classifier_inverted_repair.py` | Config for classifier-inverted transition-gate repair stages. |

Both expect a trained U-Net checkpoint and BraTS / cache paths similar to the main project (often `outputs_10hour/`).

## Roadmap

Add new YAML files for new repair ideas; leave the root `configs/` paper YAMLs untouched.
