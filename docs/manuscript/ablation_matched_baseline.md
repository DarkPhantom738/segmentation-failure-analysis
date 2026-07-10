# Manuscript notes: layer ablation (matched baseline)

Primary ablation results for the manuscript use the **matched sliding-window baseline** (375 validation cases). Source files:

- `outputs_10hour/layer_interventions/matched_baseline/ablation_summary.csv`
- `outputs_10hour/layer_interventions/matched_baseline/baseline_comparison.csv`
- `outputs_10hour/layer_interventions/matched_baseline/baseline_comparison.md`
- `outputs_10hour/layer_interventions/matched_baseline/figures/`

## Main table (mean ablation, Dice degradation)

Use **matched-baseline** values:

| Rank | Layer | Mean Dice degradation |
|------|-------|----------------------:|
| 1 | decoder1 | 0.889 |
| 2 | encoder1 | 0.498 |
| 3 | decoder2 | 0.399 |
| 4 | encoder2 | 0.273 |
| 5 | decoder3 | 0.055 |
| 6 | encoder3 | 0.027 |
| 7 | encoder4 | 0.001 |
| 8 | bottleneck | −0.000 |
| 9 | decoder4 | −0.002 |

Cohort mean baseline Dice (matched sliding-window): **0.889**.

## Robustness analysis (TTA baseline)

Scoring the same ablated masks against TTA validation predictions yields:

| Rank | Layer | TTA Dice degradation |
|------|-------|---------------------:|
| 1 | decoder1 | 0.883 |
| 2 | encoder1 | 0.493 |
| 3 | decoder2 | 0.393 |
| 4 | encoder2 | 0.267 |
| 5 | decoder3 | 0.050 |
| 6 | encoder3 | 0.022 |
| 7 | encoder4 | −0.005 |
| 8 | bottleneck | −0.006 |
| 9 | decoder4 | −0.008 |

- Layer-rank Spearman correlation (matched vs TTA): **ρ = 1.000**
- Absolute degradations differ by a uniform **+0.006** (matched − TTA)
- Top-3 unchanged: decoder1, encoder1, decoder2

Describe TTA-baseline numbers as a **robustness analysis**, not the primary result.

## Wording

Prefer **functional dependence on intact spatial activations** over “layer necessary.” Whole-layer mean ablation does not establish that any single probe direction is causally necessary.
