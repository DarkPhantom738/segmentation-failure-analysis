# Triage snapshot (`results/paper/triage_20260712/`)

Canonical nested-CV **confidence + consistency** triage run for the epoch-5 / paper checkpoint.
This is the numeric package behind the main table in the root README.

Produced by: `scripts/run_confidence_consistency_triage.py` + `configs/confidence_consistency_triage.yaml`  
Library: `src/analysis/confidence_consistency_triage.py`

---

## Read order

1. `final_report.md` — human-readable endpoint summary and pass/fail style criteria text.
2. `aggregate_metrics.csv` — method × endpoint averages (AUPRC, AUROC, Brier, capture, …).
3. `bootstrap_comparisons.csv` — paired bootstrap deltas of proposed vs baseline.
4. `fold_metrics.csv` — per outer-fold stability.
5. `heldout_predictions.csv` — case-level OOF risk scores for plotting / method_validation.
6. Figures under `figures/` for manuscript panels.

---

## File dictionary

| File | Contents |
|---|---|
| `final_report.md` | Narrative of primary/secondary endpoints and bootstrap support. |
| `aggregate_metrics.csv` | Aggregated discrimination / calibration / capture metrics by method. |
| `fold_metrics.csv` | Outer-fold breakdown (e.g. whether combined wins 5/5 folds). |
| `bootstrap_comparisons.csv` | Mean Δ and 95% CIs for key metrics (AUPRC, capture@20%, …). |
| `heldout_predictions.csv` | Case IDs with nested-CV out-of-fold scores for each method arm. |
| `case_level_features.csv` | Feature matrix actually used in this triage run (joined confidence + consistency, etc.). |
| `feature_coefficients.csv` | Logistic coefficients (where applicable) for interpretability. |
| `feature_selection_frequency.csv` | How often features enter selected models across folds. |
| `calibration_metrics.csv` | Slope/intercept / related calibration summaries. |
| `complementarity_analysis.csv` | Whether consistency adds information beyond confidence. |
| `risk_coverage_metrics.csv` | Risk–coverage / selective prediction summaries. |

### `figures/`

| Figure | Typical use |
|---|---|
| `precision_recall_curves.png` | Main PR panel (root README). |
| `failure_capture_by_review_budget.png` | Capture vs review budget. |
| `bootstrap_delta_auprc.png` | Paired bootstrap ΔAUPRC. |
| `risk_coverage_curves.png` | Selective prediction curves. |
| `calibration_plot.png` | Calibration visualization. |
| `fold_stability.png` | Outer-fold AUPRC stability. |
| `feature_selection_frequency.png` | Feature selection bar/heatmap. |

---

## Methods columns you will see

Names vary slightly by CSV, but arms generally include:

- Confidence only (primary baseline)
- Confidence + morphology
- Confidence + pooled representations
- **Confidence + representation–output consistency** (proposed)
- Sometimes additional ablations

Primary endpoint: **lowest-quality 20%** by mean foreground Dice.  
Secondary: mean foreground Dice &lt; 0.70.

---

## Relationship to method_validation/

`method_validation/` re-reads triage outputs (or equivalent paths) to produce the formal bootstrap primary table, ablations, and capture figures used for the paper’s validation writeup. Prefer that folder for “vs confidence only” manuscript packaging; prefer this folder for the full nested-CV dump.

## Roadmap

Treat as **frozen snapshot**. New triage experiments → new dated output directory under local `outputs_*`, then optional new dated folder under `results/paper/` if adopting as canonical.
