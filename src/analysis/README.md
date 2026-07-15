# Analysis (`src/analysis/`)

Library code for failure labeling, confidence features, representation–output consistency, nested-CV triage, method validation, and (separately) RQ1 probing/editing screens.

**CLIs:** [`scripts/README.md`](../../scripts/README.md) for the paper path; [`extra/scripts/README.md`](../../extra/scripts/README.md) for exploratory work.

---

## How to read this directory (paper path first)

```text
1. failure labels / case table     analyze.py  +  failure_labels.py
2. layer matrices                  layer_export.py  +  layer_io.py
3. GT-free confidence              layer_aware_latent_risk.py
                                   (+ uncertainty_metrics.py)
4. consistency gaps + feasibility  representation_output_consistency.py
5. main nested-CV triage           confidence_consistency_triage.py
6. bootstrap / ablations / figs    method_validation.py
```

Committed products of steps 3–6: [`results/paper/`](../../results/paper/README.md).

---

## Main pipeline modules (required for RQ2)

| Module | Role in detail |
|---|---|
| `analyze.py` | Builds enriched case tables from TTA export metrics (Dice, boundary/miss flags, entropy summaries). Invoked via root `analyze_failures.py`. |
| `failure_labels.py` | Shared helpers: lesion size / connected components, miss-fraction style flags, label construction used across analyses. |
| `layer_export.py` | Runs inference and writes globally pooled activations for the nine `LAYER_NAMES` stages + index CSV. Invoked via `export_layer_embeddings.py`. |
| `layer_io.py` | Load/align embedding index tables and matrices; shared bad-case label helpers for recoverability screens. |
| `uncertainty_metrics.py` | Low-level entropy-mask summaries and a small AUC helper used when relating uncertainty maps to error maps. |
| `layer_aware_latent_risk.py` | Inventory artifacts; compute **GT-free** case-level confidence features from TTA softmax/entropy; block GT-leaked columns; optional layer-aware baselines. Stage CLI: `scripts/run_layer_aware_latent_risk.py`. |
| `representation_output_consistency.py` | Core consistency idea: (1) anatomy from **predicted** mask, (2) Ridge probes on layer embeddings fit **only on train-fold GT**, (3) gap features (signed/abs/standardized/relative), (4) nested-CV feasibility vs baselines. |
| `confidence_consistency_triage.py` | **Primary scientific evaluation.** Nested CV comparing confidence alone vs +morphology / +pooled reps / +consistency / combined; calibration; bootstrap; figures; fold reports. This is what backs `results/paper/triage_20260712/`. |
| `method_validation.py` | Validation layer on frozen triage scores: paired bootstrap deltas, feature ablations, permutation importance, risk–coverage / capture curves, seed-robustness tables, manuscript-facing `validation_summary.md`. |

---

## RQ1 / exploratory modules (optional)

These are used from `extra/scripts/` and support the claim that anatomy is often **decodable** but **not reliably controllable** via probe edits.

| Module | Role |
|---|---|
| `semantic_directions.py` | Fit linear probes; lift coefficients into activation-space edit directions (GAP adjoint / bottleneck pseudoinverse notes in module docstring). |
| `representation_editing.py` | Sweep edit strengths; measure selectivity vs collateral damage; figures. |
| `layer_interventions.py` | Whole-layer mean ablation / spatial interventions; prediction quantity helpers. |
| `layer_holdout_recoverability.py` | Locked holdout recoverability heatmaps (Ridge pipelines). |
| `probe_screen_common.py` | Shared matched-random screening utilities (stratified case picks, random unit directions, ratios). |
| `edema_probe_screen.py` | Decoder1 edema-direction screen vs matched random. |
| `volume_probe_screen.py` | Decoder2 tumor-volume screen vs matched random. |

---

## Critical naming / leakage conventions

1. **Detector inputs must be GT-free at inference.** GT is allowed only to:
   - fit probes inside outer-train folds,
   - define failure labels / thresholds (often fit on train fold then applied to held-out),
   - compute evaluation metrics.
2. Columns historically named like `out_gt_edema_frac` in consistency tables can still mean **predicted-mask** anatomy in the detector path. Do not interpret the substring `gt` as “ground-truth leaked into the score.”
3. GT-linked uncertainty/error columns from failure tables are **blocked** from detector feature sets (see guards in `layer_aware_latent_risk.py` / consistency module).
4. Nested-CV structure (outer / inner folds, seeds, C grid) in triage configs is part of the frozen protocol for the published tables.

---

## Typical artifacts produced

| Artifact family | Written when | Committed snapshot |
|---|---|---|
| `failure_metrics.csv` | After TTA + `analyze_failures.py` | `results/paper/failure_metrics.csv` |
| `confidence_features.csv` | Layer-aware confidence stage | `results/paper/confidence_features.csv` |
| Consistency case features | Consistency stage | `results/paper/consistency/case_level_features.csv` |
| Triage aggregate / bootstrap / preds | Triage stage | `results/paper/triage_20260712/` |
| Validation package | Method validation | `results/paper/method_validation/` |

New runs should write under gitignored `outputs_*` unless you intentionally refresh the committed snapshot.

---

## Roadmap

| Status | Item |
|---|---|
| **Frozen** | Consistency + triage algorithms behind `results/paper/triage_20260712` |
| **Active** | Reuse same modules for converged multi-seed path (path retargeting only) |
| **Optional** | RQ1 screens / repair diagnostics via `extra/` |
| **Non-goal** | “Streamlining” by swapping sklearn loops if it risks numeric drift |

## Related docs

- Pipeline stages: [`docs/paper_pipeline.md`](../../docs/paper_pipeline.md)
- Wording cautions: [`extra/docs/reporting_notes.md`](../../extra/docs/reporting_notes.md)
- Config knobs: [`configs/README.md`](../../configs/README.md)
