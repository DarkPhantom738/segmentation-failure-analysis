# Data package (`src/data/`)

Everything between raw BraTS folders on disk and PyTorch batches / case lists used by training and inference.

---

## Files

| File | Responsibility | Used by |
|---|---|---|
| `__init__.py` | Package docstring / marker. | — |
| `brats_dataset.py` | **Core BraTS I/O.** Discovers cases, applies the paper train/val split, loads NIfTI modalities + seg, runs preprocessing, builds `DataLoader`s. Defines `BraTSCase`, `BraTSTrainingDataset` (random patches), `BraTSVolumeDataset` (full volumes for val/export). | `train.py`, trainers, embedding export, repair/editing scripts |
| `preprocessing.py` | Label remap (`4→3`), spacing resample (`scipy.ndimage.zoom`), intensity clip + z-score (optional brain mask), crop-to-nonzero, random crop for training patches. | `brats_dataset.py` |
| `converged_splits.py` | **Fixed multi-seed patient protocol.** Splits 1,251 → 876/375 with `data_split_seed`, then 876 → train/development with `development_split_seed`. Persists JSON under `outputs_converged/shared_split/` so every model seed reuses the same IDs. Helpers: `seed_output_dirname`, `case_list_hash`, `prepare_or_load_shared_splits`. | `train_converged.py`, `converged_trainer.py` |

---

## Split story (read this once)

### Paper / epoch-5 path (`ten_hour.yaml`)

```text
discover all cases with FLAIR present
  → split_cases(val_fraction=0.30, seed=42)
  → 876 train  /  375 validation
```

- Validation (375) is the cohort for TTA, failure tables, confidence, consistency, and triage nested CV.
- Documented counts: [`docs/manuscript/environment_and_split.md`](../../docs/manuscript/environment_and_split.md).

### Converged multi-seed path (`converged_unet.yaml`)

```text
same 876 / 375 outer split (shared)
  → inside 876: train / development (e.g. 788 / 88)
  → final evaluation = original 375 (held out from training dynamics)
```

Development is used for early stopping / LR plateaus only. Downstream triage for a converged seed still evaluates risk ranking on that seed’s predictions for the held-out cohort—see `scripts/run_seed_downstream.sh`.

---

## Key behaviors in `brats_dataset.py`

| Function / class | Behavior |
|---|---|
| `discover_brats_cases` | Sorted case dirs under `data_root` with `{id}_flair.nii.gz`. |
| `limit_cases` | Optional cap for smoke tests. |
| `split_cases` | Deterministic shuffle + val take. |
| `BraTSCase` | Loads 4 modalities + seg, remaps labels, resamples, normalizes, crops. |
| Training dataset | Random spatial crops for SGD. |
| Volume dataset | Full preprocessed volumes for sliding-window inference. |
| Caching | May write preprocessed tensors under a cache root from config (local `outputs_*/cache/`). |

---

## Roadmap

| Status | Item |
|---|---|
| Stable | BraTS layout + remap + paper split helpers |
| Stable | Shared converged JSON split persistence |
| Do not change casually | Split seeds / fractions if you still want `results/paper` comparability |
| Out of scope | Alternate datasets (unless a new study branch) |

## Related

- Config data blocks: `configs/ten_hour.yaml`, `configs/converged_unet.yaml`
- Expected on-disk tree: [`data/README.md`](../../data/README.md)
