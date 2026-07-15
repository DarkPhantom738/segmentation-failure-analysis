# Models (`src/models/`)

Neural network definitions. The **paper triage path** only needs `unet3d.py`. Repair modules support exploratory `extra/` experiments and are not required to reproduce the root README table.

---

## Files

| File | Role | Track |
|---|---|---|
| `__init__.py` | Package marker. | — |
| **`unet3d.py`** | Four-class 3D U-Net with instance-norm ConvBlocks, bottleneck embedding head, and hooks for the nine analysis stages. Forward: `(logits, embedding, bottleneck)`. `LAYER_NAMES` lists encoder1→decoder1 in shallow→deep→shallow order. Also: `DiceCrossEntropyLoss`, `build_model`. | **RQ2 paper (required)** |
| `spatial_edema_repair.py` | Frozen U-Net + small spatially gated module aiming to edit edema via decoder1 channel directions (probe + class-weight hybrid). | RQ1 / repair (optional) |
| `classifier_inverted_repair.py` | Classifier-inverted decoder1 repair using predicted class-transition gates. | RQ1 / repair (optional) |

---

## Why the U-Net exposes embeddings

Downstream analyses need **the same** hidden representations that the network used:

1. **Layer embedding export** (`src/analysis/layer_export.py`) pools activations at each of `LAYER_NAMES`.
2. **Consistency** fits Ridge probes on those embeddings (train-fold GT) and compares to predicted-mask anatomy.
3. **Editing / interventions** (`src/training/editing_inference.py`, `intervention_inference.py`) inject channel deltas at named layers during sliding-window inference.

Changing the architecture or `LAYER_NAMES` order would break configs and any regenerated embeddings relative to historical runs.

---

## Loss and heads (U-Net)

- Multi-class Dice + cross-entropy combination used in both short and converged training.
- Softmax over 4 classes; BraTS enhancing tumor is class index 3 after remap.
- Bottleneck spatial map → global embedding via `embedding_head` (used as one of the layer readouts / features in analysis).

---

## Roadmap

| Status | Item |
|---|---|
| Frozen for paper | `unet3d.py` topology used by epoch-5 and converged checkpoints |
| Exploratory | Spatial / classifier-inverted repair — feasibility only, not a product claim |
| Non-goal | Drop-in swap to nnU-Net / MONAI nets without a new study protocol |

## Related

- Training: [`../training/README.md`](../training/README.md)
- Layer names in analysis configs: see `configs/consistency_failure_detection.yaml` / embedding export
- Repair CLIs: [`../../extra/scripts/README.md`](../../extra/scripts/README.md)
