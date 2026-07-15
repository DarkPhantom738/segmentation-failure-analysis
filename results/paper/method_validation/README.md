# Method validation snapshot (`results/paper/method_validation/`)

Formal validation package layered on the frozen triage scores: bootstrap comparisons, feature ablations / importance, risk–coverage, capture curves, and a markdown summary suitable for Methods/Results drafting.

Produced by: `scripts/run_method_validation.py` + `configs/method_validation.yaml`  
Library: `src/analysis/method_validation.py`

Upstream triage snapshot: [`../triage_20260712/`](../triage_20260712/README.md).

---

## Read order

1. `validation_summary.md` — start here; cites the headline ΔAUPRC / capture intervals.
2. `bootstrap_primary.csv` — paired case-level bootstrap of proposed − confidence.
3. `baseline_comparison.csv` — side-by-side method metrics.
4. `feature_ablation.csv` + `feature_permutation_importance.csv` — where the gain comes from (enhancing-fraction gaps dominate).
5. Figures in `figures/` for paper panels (capture curves also linked from root README).

---

## File dictionary

| File | Contents |
|---|---|
| `validation_summary.md` | Human-readable validation report (bootstrap, ablations, caveats). |
| `bootstrap_primary.csv` | Primary endpoint bootstrap deltas + CIs + P(proposed better). |
| `baseline_comparison.csv` | Metric table across detector methods. |
| `feature_ablation.csv` | Drop-feature / bundle ablation AUPRCs. |
| `feature_permutation_importance.csv` | Mean AUPRC drop when permuting each feature (fold-aware). |
| `feature_outcome_correlations.csv` | Univariate-style associations between features and failure labels. |
| `feature_selection_frequency.csv` | Selection frequency across folds. |
| `fold_stability.csv` | Outer-fold stability of Δ metrics. |
| `risk_coverage_extended.csv` | Extended risk–coverage operating points. |
| `seed_robustness.csv` / `seed_robustness_summary.csv` | Robustness checks across evaluation seeds where configured. |
| `tumor_core_dice.csv` | Auxiliary tumor-core Dice related summaries (when generated). |

### `figures/`

| Figure | Role |
|---|---|
| `failure_capture_curves.png` | Capture vs review budget (root README). |
| `bootstrap_delta_auprc.png` | Bootstrap ΔAUPRC distribution / intervals. |
| `baseline_auprc.png` | Baseline AUPRC comparison. |
| `feature_ablation.png` | Ablation bar chart. |
| `feature_permutation_importance.png` | Importance ranking. |
| `feature_outcome_correlations.png` | Correlation overview. |
| `fold_stability.png` | Fold-wise ΔAUPRC. |
| `risk_coverage_curves.png` | Risk vs coverage. |
| `retained_dice_vs_coverage.png` | Mean Dice of retained cases vs coverage. |

---

## How this differs from triage_20260712/

| | `triage_20260712/` | `method_validation/` |
|---|---|---|
| Focus | Full nested-CV training + metrics dump | Decision/validation packaging |
| Best for | Exact OOF scores, coefficients, fold table | Manuscript bootstrap CI / ablations / figures |
| Overlap | Both discuss AUPRC and bootstrap | Prefer validation_summary for prose |

## Roadmap

Frozen with the epoch-5 paper claim. Converged-seed validation artifacts should live under that seed’s local analysis directory (or a future `results/paper/converged_*` snapshot), not overwrite this tree.
