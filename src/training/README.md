# Training & inference (`src/training/`)

Loops that optimize the U-Net, and inference utilities that turn a checkpoint into volumes, TTA confidence features, embeddings, or edited predictions.

---

## Roadmap of modules (by use)

### Required for the paper triage pipeline

| File | What it does |
|---|---|
| `trainer.py` | Original training loop for `train.py` / `ten_hour.yaml` (patch SGD, logging, checkpoints under `outputs_10hour/`). |
| `inference.py` | Sliding-window patch aggregation over a full volume. Accumulates logits with a visit-count map; returns mean patch embedding. Used by export, TTA, and editing backends. |
| `tta.py` | **8-fold axis-flip TTA**: average softmax over identity + all nonempty flip subsets; hard mask from argmax of the average. This is the source of ambient predictive entropy / confidence features used in triage. |
| `metrics.py` | Whole-tumor mask helper + Dice (binary and WT). Building block for failure tables and development metrics. |

### Required for converged multi-seed replication

| File | What it does |
|---|---|
| `converged_trainer.py` | Longer training with development-set Dice early stopping / LR schedule; writes under `outputs_converged/seed_XXX/`; never touches `outputs_10hour/` paper artifacts. |
| `converged_metrics.py` | Per-class Dice (necrosis / edema / enhancing) with explicit empty-class conventions for development reporting. |

### Optional RQ1 / repair (used from `extra/`)

| File | What it does |
|---|---|
| `layer_inference.py` | Inference helpers that expose intermediate activations for analysis. |
| `editing_inference.py` | Sliding-window forward with additive channel edits at a named layer (`alpha * direction`). |
| `intervention_inference.py` | Spatial mean / ablation-style interventions during inference. |
| `decoder_cache_inference.py` | Cache late-decoder patch maps (e.g. decoder1/decoder2) so repair/probe screens can reuse them without full re-encode. |
| `spatial_repair_trainer.py` | Train the tiny spatial edema repair module with frozen backbone. |

`__init__.py` is a package marker.

---

## Paper inference stack (how the pieces connect)

```text
checkpoint
  → sliding_window_inference (inference.py)
  → optionally wrap with STANDARD_TTA_AUGMENTATIONS (tta.py)
  → hard pred + entropy / confidence summaries
  → analyze_failures.py → failure_metrics.csv
  → export_layer_embeddings.py → pooled layer matrices
  → analysis confidence / consistency / triage
```

TTA definition is intentionally fixed (eight flips). Changing the augmentation set changes confidence features and therefore triage metrics.

---

## Converged vs “ten hour”

| Axis | `trainer.py` + `ten_hour.yaml` | `converged_trainer.py` + `converged_unet.yaml` |
|---|---|---|
| Purpose | Original paper checkpoint (epoch 5 in reported run) | Robustness: train longer with early stop |
| Split | 876 / 375 | Same outer split + internal train/dev |
| Output root | `outputs_10hour/` | `outputs_converged/seed_*/` |
| Relation to README table | **Primary numbers** | Replication / seed variance |

---

## Roadmap

| Status | Item |
|---|---|
| Frozen | TTA protocol and sliding-window aggregation used for paper CSVs |
| Active | Finish remaining converged seeds + downstream |
| Optional | Repair trainers remain exploratory |
| Do not casually replace | Inference/TTA with MONAI/etc. if bit-identical paper regen is required |

## Related

- Entry: `train.py`, `train_converged.py`
- Model: [`../models/README.md`](../models/README.md)
- Confidence extraction: `../analysis/layer_aware_latent_risk.py`
