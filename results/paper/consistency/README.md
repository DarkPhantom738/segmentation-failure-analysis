# Consistency features (`results/paper/consistency/`)

Committed snapshot of **representation–output consistency** case-level features for the 375-case validation cohort.

## Files

| File | Role |
|---|---|
| `case_level_features.csv` | One row per case. Columns include predicted-mask anatomy summaries, probe-implied anatomy (from embeddings), and **gap** features (signed / absolute / standardized / relative) used by triage when the “consistency” or “combined” feature bundles are selected. |

## How to interpret columns

- Built by `src/analysis/representation_output_consistency.py` under `configs/consistency_failure_detection.yaml` (paper paths now resolve here / through triage inputs).
- Gaps compare **representation estimates** vs **predicted-mask** measurements — not vs GT at inference.
- Historical names may contain `gt` / `out_gt_*` substrings; in the detector path these still encode predicted-mask quantities. See [`src/analysis/README.md`](../../../src/analysis/README.md).

## Downstream consumers

| Consumer | Use |
|---|---|
| `confidence_consistency_triage` | Joins consistency features with confidence (+ optional morphology / pooled reps). |
| `method_validation` | Ablations and permutation importance that attribute gain to enhancing-fraction gaps, etc. |

## Roadmap

Keep this CSV stable relative to `triage_20260712/`. Regenerations belong under local `outputs_*` until a deliberate snapshot update.
