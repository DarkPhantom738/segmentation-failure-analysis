# Representation–Output Consistency Score — Feasibility Report

**Verdict: PROMISING**

Feasibility gates (lowest-quality 20% task):
- `auprc_plus_0.05`: PASS
- `aurc_rel_10pct`: fail
- `capture20_ge_70`: PASS
- `retained80_plus_0.02`: PASS
- `mae_rel_10pct_vs_dice_probe`: PASS
- `beats_strongest_baseline`: PASS

## Answers

0. **Confidence baseline:** unavailable without GT leakage. Cached `mean_entropy_*` / `overlap_top_*` / `confident_false_negative_fraction` columns use GT error masks (`analyze.py`). Entropy maps are missing on disk. Combined model uses inconsistency + morphology only.
1. **Does inconsistency predict quality?** Inconsistency-only AUPRC=0.722 (AUROC=0.840) for lowest-20% failures vs morphology 0.726. Combined continuous MAE=0.0773.
2. **Improve over ordinary confidence?** Not evaluated — inference-time confidence features were not available without leakage.
3. **Improve over morphology?** Combined 0.782 vs morphology 0.726 (Δ=+0.056).
4. **Improve over direct Dice probe?** Combined 0.782 vs dice_probe 0.543 (Δ=+0.239). Strongest available baseline was `morphology` (AUPRC=0.726).
5. **Most informative discrepancies (mean |coef|):** relative_gap_gt_enhancing_frac (1.276), signed_gap_gt_enhancing_frac (0.546)
6. **Signed vs absolute:** signed gaps appear among top coefficients
7. **At 20% review budget:** combined captures 71.4% of lowest-20% failures (CI 60.3–78.4%).
8. **Selective risk:** AURC=0.1251; mean Dice at 80% coverage=0.854 (cohort mean=0.801).
9. **Stability:** best-layer selection consistency (top layer fraction across outer folds): dice=1.00, gt_edema_frac=1.00, log_wt_volume=0.80
10. **Result: PROMISING**

Caveats:
- Prior runs that used GT-linked entropy/error columns as “confidence” features are invalid and should be ignored.
- `pooled_repr` is a separate high-dimensional baseline and may outperform the proposed combined model; it is not the critical confidence/output/quality-probe comparator.
- Paired bootstrap CIs for AUPRC differences can include zero; treat point-estimate gains cautiously.

This is an internal nested-CV feasibility study on 375 validation cases only. No clinical benefit, external generalization, or novelty is claimed.

## Method AUPRC summary (lowest 20%)

| Method | AUPRC | AUROC | Capture@20% |
|---|---:|---:|---:|
| morphology | 0.726 | 0.866 | 0.675 |
| dice_probe | 0.543 | 0.779 | 0.519 |
| rep_anatomy | 0.540 | 0.814 | 0.558 |
| inconsistency | 0.722 | 0.840 | 0.649 |
| combined | 0.782 | 0.897 | 0.714 |
| pooled_repr | 0.811 | 0.931 | 0.740 |
