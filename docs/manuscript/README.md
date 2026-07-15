# Manuscript notes (`docs/manuscript/`)

Author/reviewer notes that back specific numbers in the paper (split sizes, compute environment).
These are **not** pipeline entry points.

## Files

| File | Purpose |
|---|---|
| [`environment_and_split.md`](environment_and_split.md) | Verified BraTS discovery/split counts (876 train / 375 val, seed 42, `val_fraction=0.30`), total discovered cases (1,251), modality completeness checks, and notes from the original 5-epoch training log. Use this when the manuscript claims those numbers. |

## How this relates to code

- Split logic: `src/data/brats_dataset.py` (`discover_brats_cases`, `split_cases`) and the paper config `configs/ten_hour.yaml`.
- Converged multi-seed protocol adds a further train/development split inside the 876: see `src/data/converged_splits.py` and `configs/converged_unet.yaml`. That does **not** replace the 375-case evaluation cohort used for triage.
- Committed fold IDs for nested CV outer folds: `results/paper/fold_assignments.csv`.

## Roadmap

| Status | Item |
|---|---|
| Done | Split + environment verification writeup |
| Pending (author) | Expand with GPU/runtime tables if a journal asks for compute reporting |
| Keep stable | Do not “approximate” case counts here; update only from re-verified logs |
