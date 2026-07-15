# Committed paper snapshot (`results/paper/`)

Compact tables and figures that support the **root README** claims. You can review these without GPU, BraTS download, or regenerating `outputs_*`.

**Do not casually overwrite** these files when experimenting; write new runs under gitignored `outputs_*` and only refresh this tree when intentionally updating the public snapshot.

---

## Start here

1. Read root [`README.md`](../../README.md) for the claim and primary table.
2. Open [`triage_20260712/final_report.md`](triage_20260712/final_report.md) and [`method_validation/validation_summary.md`](method_validation/validation_summary.md) for narrative of the nested-CV / bootstrap package.
3. Use the CSV dictionaries below when digging into columns.

Pipeline that *produced* these: [`docs/paper_pipeline.md`](../../docs/paper_pipeline.md).

---

## Top-level files

| File | Meaning |
|---|---|
| `failure_metrics.csv` | Case-level table for the **375** validation cohort: Dice / quality scores, flags used as **labels** or stratification—not detector features by themselves. Upstream: TTA export + `analyze_failures.py`. |
| `confidence_features.csv` | **GT-free** TTA confidence / entropy summaries per case. Upstream: `layer_aware_latent_risk` confidence stage. |
| `fold_assignments.csv` | Outer 5-fold case membership (seed 42 KFold protocol) used so triage folds are transparent/reproducible. |
| `consistency/` | See [`consistency/README.md`](consistency/README.md). |
| `triage_20260712/` | See [`triage_20260712/README.md`](triage_20260712/README.md). **Primary nested-CV result package.** |
| `method_validation/` | See [`method_validation/README.md`](method_validation/README.md). Bootstrap / ablations / capture / importance package. |

---

## How folders map to methods

```text
confidence_features.csv
        +
consistency/case_level_features.csv
        +
failure_metrics.csv  (labels / Dice)
        │
        ▼
triage_20260712/     ← nested CV models & primary metrics
        │
        ▼
method_validation/   ← formal comparisons vs confidence, figures for paper
```

---

## Subfolder READMEs

| Path | Contents |
|---|---|
| [`consistency/`](consistency/README.md) | Representation–output gap feature table |
| [`triage_20260712/`](triage_20260712/README.md) | Nested-CV aggregates, bootstrap, predictions, figures |
| [`method_validation/`](method_validation/README.md) | Ablations, permutation importance, capture curves, summary markdown |

---

## Roadmap

| Status | Item |
|---|---|
| Frozen | Epoch-5 triage + method_validation snapshot used in the README table |
| Optional | Add `converged_seed042/` (or multi-seed) compact summaries after local replication is complete |
| Policy | Path/config changes for regen OK; changing algorithm code to refresh numbers is out of scope unless intentional |

## Related

- Configs pointed at these inputs: [`configs/README.md`](../../configs/README.md)
- Regenerated locals: gitignored `outputs_confidence_consistency_triage*`, `outputs_method_validation`, etc.
