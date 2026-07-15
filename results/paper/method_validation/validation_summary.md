# Method validation summary

**Algorithm:** frozen confidence + representation–output consistency  
**Canonical triage artifacts:** `outputs_confidence_consistency_triage_20260712_030902`  
**Verdict: STRONG**

No U-Net retraining. No probe/edit/repair regeneration. Classifier-only refits used only for missing baselines (`dice_probe`, `pooled`), ablations, and seed checks.

---

## 1. Complete baseline comparison

Identical outer folds (seed 42, 5×75). Metrics from frozen held-out scores where available; `dice_probe` / `pooled` scores reused from the consistency experiment on the same folds (fail labels agree).

### lowest20_mean_fg

| Method | AUPRC | AUROC | Cap@10 | Cap@20 | Cap@30 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Confidence only | 0.805 | 0.931 | 0.442 | 0.714 | 0.870 | 0.102 |
| Morphology only | 0.726 | 0.866 | 0.403 | 0.675 | 0.766 | 0.137 |
| Confidence + morphology | 0.851 | 0.954 | 0.442 | 0.753 | 0.948 | 0.077 |
| Direct Dice probe | 0.543 | 0.779 | 0.312 | 0.519 | 0.623 | 0.208 |
| Pooled representation | 0.811 | 0.931 | 0.442 | 0.740 | 0.857 | 0.091 |
| Confidence + pooled | 0.854 | 0.948 | 0.468 | 0.740 | 0.896 | 0.081 |
| Consistency only | 0.725 | 0.844 | 0.403 | 0.649 | 0.740 | 0.127 |
| Confidence + consistency (proposed) | 0.895 | 0.960 | 0.468 | 0.792 | 0.961 | 0.064 |
| Confidence + morphology + consistency | 0.882 | 0.958 | 0.455 | 0.792 | 0.948 | 0.069 |

### mean_fg_lt_0.70

| Method | AUPRC | AUROC | Cap@10 | Cap@20 | Cap@30 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Confidence only | 0.770 | 0.929 | 0.425 | 0.712 | 0.863 | 0.100 |
| Morphology only | 0.732 | 0.874 | 0.397 | 0.699 | 0.795 | 0.131 |
| Confidence + morphology | 0.851 | 0.960 | 0.452 | 0.781 | 0.959 | 0.075 |
| Direct Dice probe | 0.517 | 0.767 | 0.315 | 0.507 | 0.603 | 0.210 |
| Pooled representation | 0.808 | 0.935 | 0.466 | 0.740 | 0.890 | 0.091 |
| Confidence + pooled | 0.827 | 0.951 | 0.479 | 0.781 | 0.932 | 0.078 |
| Consistency only | 0.723 | 0.844 | 0.438 | 0.658 | 0.753 | 0.127 |
| Confidence + consistency (proposed) | 0.887 | 0.973 | 0.479 | 0.822 | 0.986 | 0.064 |
| Confidence + morphology + consistency | 0.867 | 0.964 | 0.479 | 0.822 | 0.973 | 0.070 |

### mean_fg_lt_0.80

| Method | AUPRC | AUROC | Cap@10 | Cap@20 | Cap@30 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Confidence only | 0.878 | 0.930 | 0.303 | 0.557 | 0.779 | 0.097 |
| Morphology only | 0.784 | 0.834 | 0.287 | 0.533 | 0.656 | 0.160 |
| Confidence + morphology | 0.913 | 0.951 | 0.303 | 0.590 | 0.795 | 0.085 |
| Direct Dice probe | 0.696 | 0.781 | 0.279 | 0.484 | 0.566 | 0.209 |
| Pooled representation | 0.853 | 0.897 | 0.295 | 0.549 | 0.738 | 0.122 |
| Confidence + pooled | 0.839 | 0.901 | 0.279 | 0.557 | 0.754 | 0.108 |
| Consistency only | 0.727 | 0.786 | 0.287 | 0.467 | 0.590 | 0.172 |
| Confidence + consistency (proposed) | 0.896 | 0.937 | 0.303 | 0.582 | 0.779 | 0.091 |
| Confidence + morphology + consistency | 0.920 | 0.954 | 0.311 | 0.590 | 0.795 | 0.081 |

### edema_lt_0.70

| Method | AUPRC | AUROC | Cap@10 | Cap@20 | Cap@30 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Confidence only | 0.849 | 0.939 | 0.411 | 0.667 | 0.844 | 0.098 |
| Morphology only | 0.767 | 0.892 | 0.367 | 0.633 | 0.800 | 0.130 |
| Confidence + morphology | 0.896 | 0.958 | 0.400 | 0.722 | 0.867 | 0.080 |
| Direct Dice probe | 0.655 | 0.804 | 0.344 | 0.578 | 0.678 | 0.200 |
| Pooled representation | 0.679 | 0.844 | 0.344 | 0.578 | 0.689 | 0.149 |
| Confidence + pooled | 0.808 | 0.916 | 0.356 | 0.644 | 0.822 | 0.099 |
| Consistency only | 0.645 | 0.797 | 0.289 | 0.522 | 0.711 | 0.163 |
| Confidence + consistency (proposed) | 0.854 | 0.939 | 0.389 | 0.700 | 0.844 | 0.091 |
| Confidence + morphology + consistency | 0.847 | 0.947 | 0.389 | 0.689 | 0.867 | 0.089 |

### lowest20_edema

| Method | AUPRC | AUROC | Cap@10 | Cap@20 | Cap@30 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Confidence only | 0.794 | 0.921 | 0.442 | 0.701 | 0.831 | 0.107 |
| Morphology only | 0.727 | 0.864 | 0.403 | 0.675 | 0.805 | 0.130 |
| Confidence + morphology | 0.823 | 0.930 | 0.468 | 0.714 | 0.857 | 0.091 |
| Direct Dice probe | 0.596 | 0.807 | 0.338 | 0.597 | 0.714 | 0.167 |
| Pooled representation | 0.483 | 0.789 | 0.273 | 0.481 | 0.649 | 0.166 |
| Confidence + pooled | 0.731 | 0.875 | 0.403 | 0.701 | 0.766 | 0.114 |
| Consistency only | 0.618 | 0.785 | 0.364 | 0.545 | 0.727 | 0.160 |
| Confidence + consistency (proposed) | 0.779 | 0.901 | 0.403 | 0.701 | 0.844 | 0.105 |
| Confidence + morphology + consistency | 0.799 | 0.920 | 0.442 | 0.727 | 0.857 | 0.093 |

## 2. Feature importance

Held-out permutation importance (shuffle one consistency column at a time through the trained confidence+consistency logistic per outer fold).

| Rank | Feature | Mean AUPRC drop |
|---:|---|---:|
| 1 | `relative_gap_gt_enhancing_frac` | 0.2572 |
| 2 | `signed_gap_gt_enhancing_frac` | 0.0220 |
| 3 | `signed_gap_gt_compactness` | 0.0181 |
| 4 | `relative_gap_log_wt_volume` | 0.0104 |
| 5 | `absolute_gap_gt_compactness` | 0.0084 |
| 6 | `standardized_gap_gt_necrosis_frac` | 0.0055 |
| 7 | `absolute_gap_gt_edema_frac` | 0.0014 |
| 8 | `standardized_gap_gt_compactness` | 0.0013 |
| 9 | `relative_gap_gt_edema_frac` | 0.0003 |
| 10 | `signed_gap_gt_necrosis_frac` | -0.0000 |
| 11 | `standardized_gap_gt_enhancing_frac` | -0.0020 |
| 12 | `absolute_gap_gt_necrosis_frac` | -0.0020 |

## 3. Feature ablation

Each row = confidence features + listed consistency subset; logistic re-fit on identical outer folds.

| Ablation | AUPRC | Cap@20 |
|---|---:|---:|
| drop_gt_edema_frac | 0.896 | 0.779 |
| drop_gt_necrosis_frac | 0.889 | 0.779 |
| tissue_fraction | 0.887 | 0.792 |
| all_gaps | 0.886 | 0.766 |
| drop_gt_compactness | 0.886 | 0.805 |
| drop_boundary_complexity | 0.886 | 0.805 |
| drop_log_wt_volume | 0.885 | 0.805 |
| relative_only | 0.879 | 0.805 |
| signed_only | 0.811 | 0.701 |
| confidence_only | 0.805 | 0.714 |
| shape_boundary | 0.805 | 0.727 |
| volume_related | 0.803 | 0.727 |
| absolute_only | 0.778 | 0.714 |
| drop_gt_enhancing_frac | 0.773 | 0.701 |

### Still works after removing…?

- **whole-tumor volume:** AUPRC=0.885 (vs confidence 0.805) → yes
- **enhancing fraction:** AUPRC=0.773 (vs confidence 0.805) → marginal/no
- **edema fraction:** AUPRC=0.896 (vs confidence 0.805) → yes
- **compactness / boundary-complexity proxy:** AUPRC=0.886 (vs confidence 0.805) → yes
- **necrosis fraction:** AUPRC=0.889 (vs confidence 0.805) → yes

## 4. Fold stability

- `lowest20_mean_fg`: proposed wins **5/5** folds
- `mean_fg_lt_0.70`: proposed wins **5/5** folds

| Failure | Fold | ΔAUPRC |
|---|---:|---:|
| lowest20_mean_fg | 0 | +0.072 |
| lowest20_mean_fg | 1 | +0.211 |
| lowest20_mean_fg | 2 | +0.059 |
| lowest20_mean_fg | 3 | +0.003 |
| lowest20_mean_fg | 4 | +0.040 |
| mean_fg_lt_0.70 | 0 | +0.075 |
| mean_fg_lt_0.70 | 1 | +0.151 |
| mean_fg_lt_0.70 | 2 | +0.060 |
| mean_fg_lt_0.70 | 3 | +0.003 |
| mean_fg_lt_0.70 | 4 | +0.086 |

## 5. Seed stability

Classifier-only retrain; frozen features.

| Failure | Method | Metric | Mean | Std | CV |
|---|---|---|---:|---:|---:|
| lowest20_mean_fg | conf_consistency | auprc | 0.886 | 0.0000 | 0.0000 |
| lowest20_mean_fg | conf_consistency | capture_at_20 | 0.766 | 0.0000 | 0.0000 |
| lowest20_mean_fg | confidence | auprc | 0.810 | 0.0000 | 0.0000 |
| lowest20_mean_fg | confidence | capture_at_20 | 0.740 | 0.0000 | 0.0000 |
| mean_fg_lt_0.70 | conf_consistency | auprc | 0.900 | 0.0000 | 0.0000 |
| mean_fg_lt_0.70 | conf_consistency | capture_at_20 | 0.822 | 0.0000 | 0.0000 |
| mean_fg_lt_0.70 | confidence | auprc | 0.805 | 0.0000 | 0.0000 |
| mean_fg_lt_0.70 | confidence | capture_at_20 | 0.712 | 0.0000 | 0.0000 |

## 6. Bootstrap comparisons

Paired case-level bootstrap, n=5000.

| Failure | Metric | Δ | 95% CI | P(proposed better) |
|---|---|---:|---|---:|
| lowest20_mean_fg | auprc | +0.0898 | [+0.0305, +0.1560] | 0.998 |
| lowest20_mean_fg | capture20 | +0.0870 | [+0.0111, +0.1667] | 0.975 |
| lowest20_mean_fg | aurc | +0.0008 | [-0.0056, +0.0082] | 0.440 |
| mean_fg_lt_0.70 | auprc | +0.1164 | [+0.0575, +0.1861] | 1.000 |
| mean_fg_lt_0.70 | capture20 | +0.0966 | [+0.0154, +0.1781] | 0.988 |
| mean_fg_lt_0.70 | aurc | +0.0015 | [-0.0043, +0.0071] | 0.303 |

For AURC, lower is better; `frac_proposed_better` is P(ΔAURC < 0).

## 7. Risk–coverage

See `figures/risk_coverage_curves.png` and `risk_coverage_extended.csv`.

| Method | AURC | Dice@70% | Dice@80% | Dice@90% |
|---|---:|---:|---:|---:|
| Confidence only | 0.1149 | 0.876 | 0.858 | 0.832 |
| Confidence + consistency (proposed) | 0.1159 | 0.879 | 0.862 | 0.834 |
| Confidence + morphology | 0.1109 | 0.880 | 0.860 | 0.835 |
| Confidence + pooled | 0.1120 | 0.875 | 0.858 | 0.838 |

## 8. Failure capture

See `figures/failure_capture_curves.png`.

| Method | Cap@10 | Cap@20 | Cap@30 |
|---|---:|---:|---:|
| Confidence only | 0.442 | 0.714 | 0.870 |
| Morphology only | 0.403 | 0.675 | 0.766 |
| Confidence + morphology | 0.442 | 0.753 | 0.948 |
| Direct Dice probe | 0.312 | 0.519 | 0.623 |
| Pooled representation | 0.442 | 0.740 | 0.857 |
| Confidence + pooled | 0.468 | 0.740 | 0.896 |
| Consistency only | 0.403 | 0.649 | 0.740 |
| Confidence + consistency (proposed) | 0.468 | 0.792 | 0.961 |
| Confidence + morphology + consistency | 0.455 | 0.792 | 0.948 |

## 9. Clinical / mechanistic interpretation

### Why overall failures improve

Top Spearman correlations of consistency gaps with **mean foreground Dice**:

- `relative_gap_gt_enhancing_frac` ↔ mean_fg_dice: ρ=-0.300
- `standardized_gap_gt_compactness` ↔ mean_fg_dice: ρ=-0.257
- `absolute_gap_gt_compactness` ↔ mean_fg_dice: ρ=-0.256
- `relative_gap_gt_compactness` ↔ mean_fg_dice: ρ=-0.227
- `signed_gap_gt_compactness` ↔ mean_fg_dice: ρ=+0.190

Enhancing-fraction and volume mismatches track global segmentation quality and tumor-core / whole-tumor errors. Absolute and relative gaps on `gt_enhancing_frac` dominate permutation importance and leave-one-target ablation (dropping enhancing-fraction gaps hurts most).

### Why edema-specific failures do not improve

On `edema_lt_0.70`, ΔAUPRC(proposed − confidence) = +0.006. Confidence + morphology often matches or beats confidence + consistency for edema targets.

Top correlations with **edema Dice**:

- `signed_gap_gt_compactness` ↔ edema_dice: ρ=+0.326
- `absolute_gap_gt_compactness` ↔ edema_dice: ρ=-0.233
- `standardized_gap_gt_compactness` ↔ edema_dice: ρ=-0.233
- `relative_gap_gt_compactness` ↔ edema_dice: ρ=-0.207
- `absolute_gap_gt_enhancing_frac` ↔ edema_dice: ρ=-0.198

Enhancing-fraction gaps correlate more strongly with overall/mean-fg quality (mean |ρ|≈0.162) than with edema Dice (mean |ρ|≈0.152). The consistency features therefore preferentially flag composition / core–enhancement disagreements that drive mean-fg failures, not edema-boundary failures that confidence/morphology already capture.

### What failure types are detected

- **Detected well:** low mean-fg Dice / severe overall failures, often tied to enhancing-core and whole-tumor volume disagreement.
- **Detected weakly:** edema-compartment failures without global collapse.
- **Not a Dice oracle:** direct Dice probe remains weak (AUPRC≈0.54 on lowest 20%); consistency adds anatomical disagreement, not a second Dice predictor.

## 10. Verdict

**STRONG**

- Primary lowest-20% ΔAUPRC = +0.090 (bootstrap [+0.031, +0.156], P(better)=1.00)
- Primary Dice<0.70 ΔAUPRC = +0.117 (bootstrap [+0.057, +0.186], P(better)=1.00)
- Fold wins: 5/5 and 5/5
- Capture@20 improves on primary lowest-20% (0.792 vs 0.714)

Edema-specific performance remains a limitation; do not claim compartment-universal triage.

## Task 7 (TTA disagreement)

Skipped — see `tta_skipped.md`.

No novelty, deployment, or external-generalization claims.

