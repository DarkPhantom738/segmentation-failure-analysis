# Committed paper snapshot

Canonical 5-epoch triage / validation tables and figures referenced by the root README.

| Path | Contents |
|------|----------|
| `triage_20260712/` | Nested-CV aggregate metrics, bootstrap comparisons, PR curves |
| `method_validation/` | Bootstrap primary endpoint, ablations, capture curves |
| `converged_seed042/` | Compact converged-model replication snapshot and percentile-sweep plot |
| `confidence_features.csv` | Case-level TTA confidence features |
| `failure_metrics.csv` | Case list / Dice labels for the 375-case cohort |
| `consistency/case_level_features.csv` | Representation–output consistency features |
| `fold_assignments.csv` | Outer 5-fold assignments (seed 42, matches historical KFold) |

Full regenerations write to local `outputs_*` (gitignored). Do not overwrite these committed files casually.
