# Baseline comparisons (leakage-safe nested CV)

Same outer folds as consistency experiment (seed 42). Confidence is the new GT-free TTA summaries. `combined` = inconsistency + morphology (prior definition).

## Lowest-20% mean-fg Dice (n_pos≈77)

| Method | AUPRC | Capture@20% | AURC |
|---|---:|---:|---:|
| combined_conf | 0.884 | 0.792 | 0.1118 |
| combined_pooled | 0.836 | 0.792 | 0.1148 |
| pooled | 0.811 | 0.740 | 0.1152 |
| confidence | 0.806 | 0.714 | 0.1148 |
| combined | 0.791 | 0.688 | 0.1257 |
| combined_conf_pooled | 0.751 | 0.779 | 0.1232 |
| morphology | 0.726 | 0.675 | 0.1299 |
| inconsistency | 0.711 | 0.649 | 0.1471 |

## Key paired AUPRC differences (lowest20)

- **combined − confidence**: Δ=-0.014 [-0.107, +0.076] (P(a>b)=0.38; CI includes 0)
- **combined_conf − combined_pooled**: Δ=+0.047 [-0.013, +0.112] (P(a>b)=0.94; CI includes 0)
- **combined_conf − combined**: Δ=+0.092 [+0.031, +0.155] (P(a>b)=1.00; CI excludes 0)
- **combined_pooled − combined**: Δ=+0.045 [-0.032, +0.123] (P(a>b)=0.88; CI includes 0)
- **pooled − combined**: Δ=+0.020 [-0.068, +0.106] (P(a>b)=0.69; CI includes 0)
