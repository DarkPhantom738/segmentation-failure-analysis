# Confidence + Consistency Triage — Final Report

**Verdict: STRONG**

Feasibility / success gates:
- `both_primary_delta_ge_0.03`: PASS
- `bootstrap_excl0_at_least_one_and_positive_other`: PASS
- `fold_wins_ge_4_both`: PASS
- `capture_or_rc_improves`: PASS
- `both_primary_positive`: PASS
- `one_delta_ge_0.02`: PASS
- `bootstrap_mostly_positive`: PASS
- `fold_wins_ge_3_both`: PASS
- `perm_near_chance`: PASS

## Answers

1. **Does consistency improve confidence for lowest-20%?** ΔAUPRC=+0.087 (conf+cons=0.893 vs conf=0.806)
2. **Does it improve confidence for Dice < 0.70?** ΔAUPRC=+0.127 (conf+cons=0.889 vs conf=0.761)
3. **Bootstrap support?** lowest20_mean_fg: Δ=+0.087 [+0.032,+0.146] P(better)=1.00; mean_fg_lt_0.70: Δ=+0.125 [+0.064,+0.198] P(better)=1.00
4. **Capture@20%?** conf+cons=0.779 vs conf=0.714
5. **Risk–coverage?** AURC conf+cons=0.1144 vs conf=0.1148; Dice@80% 0.861 vs 0.858
6. **Fold/seed stability?** lowest20_mean_fg: wins 5/5 meanΔ=+0.072 [+0.003,+0.180]; mean_fg_lt_0.70: wins 5/5 meanΔ=+0.076 [+0.003,+0.151]; logistic seeds AUPRC=42:0.886,123:0.886,2026:0.886,42:0.959,123:0.959,2026:0.959
7. **Which discrepancies help?** relative_gap_gt_enhancing_frac (3.419), signed_gap_gt_enhancing_frac (1.780), signed_gap_gt_edema_frac (1.138), relative_gap_gt_compactness (1.095), relative_gap_log_wt_volume (0.977)
8. **Does morphology explain it?** conf+morph AUPRC=0.848 vs conf+cons=0.893 on lowest20; morphology does not fully explain the gain
9. **How does conf+pooled compare?** conf+pooled=0.854 vs confidence=0.806 vs conf+cons=0.893; pooled can beat confidence alone but underperforms compact anatomical gaps (noise/collinearity in high-dim pooled features; no explicit representation–output disagreement).
10. **Verdict** STRONG

Permutation check: {'analysis': 'label_permutation', 'auprc': 0.1944678736228172, 'chance_baseline': 0.2053333333333333, 'model_seed': 42, 'detail': nan}

No clinical-deployment, external-generalization, or novelty claims.

## Aggregate AUPRC (selected methods)

| Failure | confidence | conf+cons | conf+morph | conf+pooled | consistency |
|---|---:|---:|---:|---:|---:|
| lowest20_mean_fg | 0.806 | 0.893 | 0.848 | 0.854 | 0.725 |
| mean_fg_lt_0.80 | 0.875 | 0.895 | 0.911 | 0.879 | 0.734 |
| mean_fg_lt_0.70 | 0.761 | 0.889 | 0.847 | 0.827 | 0.724 |
| edema_lt_0.70 | 0.849 | 0.855 | 0.896 | 0.761 | 0.642 |
| lowest20_edema | 0.794 | 0.778 | 0.823 | 0.701 | 0.618 |
