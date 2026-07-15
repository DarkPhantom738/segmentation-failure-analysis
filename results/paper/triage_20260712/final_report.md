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

1. **Does consistency improve confidence for lowest-20%?** ΔAUPRC=+0.090 (conf+cons=0.895 vs conf=0.805)
2. **Does it improve confidence for Dice < 0.70?** ΔAUPRC=+0.117 (conf+cons=0.887 vs conf=0.770)
3. **Bootstrap support?** lowest20_mean_fg: Δ=+0.089 [+0.031,+0.152] P(better)=1.00; mean_fg_lt_0.70: Δ=+0.115 [+0.056,+0.185] P(better)=1.00
4. **Capture@20%?** conf+cons=0.792 vs conf=0.714
5. **Risk–coverage?** AURC conf+cons=0.1159 vs conf=0.1149; Dice@80% 0.862 vs 0.858
6. **Fold/seed stability?** lowest20_mean_fg: wins 5/5 meanΔ=+0.077 [+0.003,+0.211]; mean_fg_lt_0.70: wins 5/5 meanΔ=+0.075 [+0.003,+0.151]
7. **Which discrepancies help?** relative_gap_gt_enhancing_frac (2.794), signed_gap_gt_enhancing_frac (1.454), signed_gap_gt_edema_frac (0.906), relative_gap_log_wt_volume (0.797), relative_gap_gt_compactness (0.774)
8. **Does morphology explain it?** conf+morph AUPRC=0.851 vs conf+cons=0.895 on lowest20; morphology does not fully explain the gain
9. **How does conf+pooled compare?** conf+pooled=0.854 vs confidence=0.805 vs conf+cons=0.895; pooled can beat confidence alone but underperforms compact anatomical gaps (noise/collinearity in high-dim pooled features; no explicit representation–output disagreement).
10. **Verdict** STRONG

Permutation check: {'analysis': 'label_permutation', 'auprc': 0.1883689978472264, 'chance_baseline': 0.20533333333333334, 'model_seed': 42}

No clinical-deployment, external-generalization, or novelty claims.

## Aggregate AUPRC (selected methods)

| Failure | confidence | conf+cons | conf+morph | conf+pooled | consistency |
|---|---:|---:|---:|---:|---:|
| lowest20_mean_fg | 0.805 | 0.895 | 0.851 | 0.854 | 0.725 |
| mean_fg_lt_0.80 | 0.878 | 0.896 | 0.913 | 0.839 | 0.727 |
| mean_fg_lt_0.70 | 0.770 | 0.887 | 0.851 | 0.827 | 0.723 |
| edema_lt_0.70 | 0.849 | 0.854 | 0.896 | 0.808 | 0.645 |
| lowest20_edema | 0.794 | 0.779 | 0.823 | 0.731 | 0.618 |
