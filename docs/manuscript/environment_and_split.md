# Dataset split and compute environment

Verified from the original 10-hour / full-dataset training log and the local BraTS archive used for that run. Numbers below are not estimates.

## Case discovery and split

**Source log:** `outputs_10hour/run_5epoch_pipeline.log` (header):

```text
=== 5-epoch full-dataset training, 30% validation ===
Tue Jul  7 23:05:54 PDT 2026
Using device: cpu
Cases — train: 876, val: 375
```

The same `Cases — train: 876, val: 375` line appears twice in that log (training start and a later resume/export entry). The log does **not** print an explicit “discovered 1251” line; `train.py` only prints the post-split counts:

```python
print(f"Cases — train: {len(train_cases)}, val: {len(val_cases)}")
```

**Implied total:** 876 + 375 = **1,251**.

**Independent re-check** (same archive path and config: `val_fraction=0.30`, `seed=42`):

| Quantity | Value |
|----------|------:|
| `discover_brats_cases(...)` | 1,251 |
| Training cases | 876 |
| Validation cases | 375 |
| Sum | 1,251 |

Discovery requires a case directory containing `{case_id}_flair.nii.gz` (`src/data/brats_dataset.py`).

## Missing / corrupt modality or segmentation

**In the training log:** zero hits for `Missing modality`, `FileNotFoundError`, `Traceback`, `Error loading`, `corrupt`, `failed to load`, or `Skipping`.

**On disk (all 1,251 discovered cases):** every case has all four modalities (`t1`, `t1ce`, `t2`, `flair`) and `{case_id}_seg.nii.gz` (0 missing paths).

**Loader behavior** (`BraTSCase.load`):

- Requires each configured modality file; if absent, raises `FileNotFoundError(f"Missing modality file: {path}")`.
- Loads `{case_id}_seg.nii.gz` via `nib.load` (raises if missing/unreadable).
- Does **not** skip, exclude, or impute missing files.

Config modalities for this run: `["t1", "t1ce", "t2", "flair"]` (`configs/ten_hour.yaml`).

## Software and hardware (final analysis environment)

Recorded from the project `.venv` and host used for the final probing / editing / ablation runs (2026-07-10).

### Operating system and hardware

| Item | Value |
|------|-------|
| OS | macOS 15.7.3 (Build 24G419), Darwin 24.6.0 |
| Machine | MacBook Pro (MacBookPro18,3) |
| Chip | Apple M1 Pro (arm64), 10 cores (8 performance + 2 efficiency) |
| Memory | 32 GB |
| Accelerator | Apple MPS available; CUDA not available |

### Python and packages (installed versions)

`requirements.txt` lists **lower bounds only** (e.g. `torch>=2.0.0`). Exact installed versions in `.venv`:

| Package | Installed |
|---------|-----------|
| Python | 3.11.9 |
| torch | 2.12.1 |
| numpy | 2.4.6 |
| nibabel | 5.4.2 |
| scipy | 1.17.1 |
| PyYAML | 6.0.3 |
| tqdm | 4.68.4 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| umap-learn | 0.5.12 |
| matplotlib | 3.11.0 |

### Device note for the original training log

The full-dataset training log header reports `Using device: cpu`. Later analysis scripts (editing, ablation, probe screens) typically selected MPS when available (`Using device: mps` in those run logs). Report training and analysis devices separately if needed.
