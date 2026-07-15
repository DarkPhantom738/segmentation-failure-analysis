# Data directory (`data/`)

Local BraTS 2021 tree expected by configs (typically `data.root: data/BraTS2021_Training`).
**Patient MRI volumes are not part of the GitHub release.** This folder may exist on a developer machine after download; cloning the repo alone does not populate it.

## Expected layout

```text
data/
  BraTS2021_Training/
    BraTS2021_XXXXX/
      BraTS2021_XXXXX_t1.nii.gz
      BraTS2021_XXXXX_t1ce.nii.gz
      BraTS2021_XXXXX_t2.nii.gz
      BraTS2021_XXXXX_flair.nii.gz
      BraTS2021_XXXXX_seg.nii.gz
```

Discovery rule (see `src/data/brats_dataset.py`): a subdirectory counts as a case only if `{case_id}_flair.nii.gz` exists.

## How the code uses this path

| Consumer | Role |
|---|---|
| `train.py` / `configs/ten_hour.yaml` | Original 876/375 split; training + TTA export |
| `train_converged.py` / `configs/converged_unet.yaml` | Shared patient splits + per-seed training |
| Embedding / consistency / repair scripts | Load volumes for inference when regenerating from checkpoint |

Many analysis stages can start from **committed CSVs** under `results/paper/` (plus local caches under gitignored `outputs_*`) without re-reading every NIfTI.

## Governance

- Obtain BraTS 2021 under [Synapse terms](https://www.synapse.org/#!Synapse:syn27046444).
- Do not commit NIfTI volumes, caches, or case IDs that re-identify subjects beyond the public BraTS naming convention already used in tables.

## Roadmap

| Status | Item |
|---|---|
| Local only | Raw BraTS on disk |
| Never commit | Volumes, preprocessed tensors, softmax stacks |
| Readers without data | Use `results/paper/` tables/figures; skip GPU regeneration |
