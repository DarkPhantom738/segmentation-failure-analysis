# Ablation baseline comparison

Compares mean Dice degradation (mean ablation) when ablated outputs are scored against:
1. TTA validation predictions from `failure_metrics.csv`
2. Matched sliding-window inference (same path/overlap as ablations, no TTA)

## Per-layer Dice degradation (mean ablation)

| Layer | TTA baseline deg. | Matched baseline deg. | Δ deg. | TTA rank | Matched rank |
|-------|------------------:|----------------------:|-------:|---------:|-------------:|
| decoder1 | 0.883 | 0.889 | +0.006 | 1 | 1 |
| encoder1 | 0.493 | 0.498 | +0.006 | 2 | 2 |
| decoder2 | 0.393 | 0.399 | +0.006 | 3 | 3 |
| encoder2 | 0.267 | 0.273 | +0.006 | 4 | 4 |
| decoder3 | 0.050 | 0.055 | +0.006 | 5 | 5 |
| encoder3 | 0.022 | 0.027 | +0.006 | 6 | 6 |
| encoder4 | -0.005 | 0.001 | +0.006 | 7 | 7 |
| bottleneck | -0.006 | -0.000 | +0.006 | 8 | 8 |
| decoder4 | -0.008 | -0.002 | +0.006 | 9 | 9 |

## Ranking stability

- Top-3 layers (TTA baseline): decoder1, encoder1, decoder2
- Top-3 layers (matched baseline): decoder1, encoder1, decoder2
- Spearman correlation of layer ranks: 1.000
- Reference baseline Dice (cohort mean): TTA = 0.883, matched sliding-window = 0.889