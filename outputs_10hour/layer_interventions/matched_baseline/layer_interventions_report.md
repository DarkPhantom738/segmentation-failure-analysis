# Layer Intervention Report (Part 1 — Ablation)

**Cases:** 375  
**Interventions:** zero, mean, noise (per-layer activation replacement)  
**Baseline:** matched sliding-window inference (same overlap/patch grid as ablations; no TTA, no ablation)

## Ablation semantics

| Layer | Intervention type |
|-------|-------------------|
| encoder1–encoder4 | **Skip ablation** — encoder runs normally; only the skip tensor at decode fusion is replaced |
| bottleneck | Replace bottleneck map, then decode |
| decoder4–decoder1 | Replace decoder block output, continue downstream decoding |

**Interpretation:** mean ablation is the primary read (removes case-specific spatial detail while preserving scale). Zero and noise are sanity checks. Compare layers using `intervention_strength_summary.csv` (degradation per unit ρ) — raw degradation alone is not comparable across layers. Results measure **functional dependence on intact spatial activations**, not necessity of any single probe direction.

Positive `mean_degradation` means the metric got worse after ablation.

**Primary baseline:** matched sliding-window inference (same overlap/patch grid as ablations; no TTA). Scoring against TTA validation predictions is a robustness analysis (`baseline_comparison.md`); layer ranks agree with Spearman ρ = 1.000.

## Perturbation strength (ρ = ||A - A'|| / ||A||)

| Layer | zero ρ | mean ρ | noise ρ | Dice deg/ρ (mean) |
|-------|--------|--------|---------|-------------------|
| encoder1 | 1.000 | 0.806 | 1.284 | 0.619 |
| encoder2 | 1.000 | 0.829 | 1.299 | 0.329 |
| encoder3 | 1.000 | 0.910 | 1.352 | 0.030 |
| encoder4 | 1.000 | 0.917 | 1.358 | 0.001 |
| bottleneck | 1.000 | 0.859 | 1.323 | -0.000 |
| decoder4 | 1.000 | 0.689 | 1.215 | -0.003 |
| decoder3 | 1.000 | 0.849 | 1.312 | 0.065 |
| decoder2 | 1.000 | 0.839 | 1.305 | 0.475 |
| decoder1 | 1.000 | 0.805 | 1.284 | 1.105 |

## Top degradations — zero ablation


### encoder1

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 7864.4213 |
| 2 | False-negative voxels | 7268.2507 |
| 3 | Boundary complexity | 1.8313 |

### encoder2

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 14109.8187 |
| 2 | False-positive voxels | 13550.9680 |
| 3 | Boundary complexity | 7.0048 |

### encoder3

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 735.7920 |
| 2 | False-positive voxels | 418.7067 |
| 3 | False-negative voxels | 16.1893 |

### encoder4

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 182.2907 |
| 2 | False-negative voxels | 57.5840 |
| 3 | Boundary complexity | 0.0830 |

### bottleneck

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 18.9733 |
| 2 | False-negative voxels | 2.9307 |
| 3 | Boundary complexity | 0.0100 |

### decoder4

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 174.5573 |
| 2 | False-negative voxels | 68.0693 |
| 3 | Boundary complexity | 0.0654 |

### decoder3

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 2393.4000 |
| 2 | False-negative voxels | 1691.4773 |
| 3 | Boundary complexity | 0.5510 |

### decoder2

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 22542.3387 |
| 2 | False-positive voxels | 21976.1707 |
| 3 | Boundary complexity | 9.4725 |

### decoder1

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 11640.9253 |
| 2 | False-negative voxels | 10604.1813 |
| 3 | Boundary complexity | 6.1747 |

## Cross-layer comparison (zero ablation, mean degradation)

| Layer | Dice | Volume | Edema frac | Enh frac | Boundary | FP | FN |
|-------|------|--------|------------|----------|----------|----|----|
| encoder1 | 0.497 | 7864.421 | 0.513 | 0.149 | 1.831 | -507.920 | 7268.251 |
| encoder2 | 0.346 | 14109.819 | 0.138 | 0.085 | 7.005 | 13550.968 | -179.560 |
| encoder3 | 0.024 | 735.792 | 0.029 | 0.014 | 0.279 | 418.707 | 16.189 |
| encoder4 | -0.000 | 182.291 | 0.005 | 0.003 | 0.083 | -63.272 | 57.584 |
| bottleneck | 0.000 | 18.973 | 0.001 | 0.000 | 0.010 | -2.261 | 2.931 |
| decoder4 | -0.002 | 174.557 | 0.005 | 0.003 | 0.065 | -97.293 | 68.069 |
| decoder3 | 0.053 | 2393.400 | 0.044 | 0.027 | 0.551 | -697.448 | 1691.477 |
| decoder2 | 0.412 | 22542.339 | 0.159 | 0.098 | 9.472 | 21976.171 | -566.168 |
| decoder1 | 0.889 | 11640.925 | 0.618 | 0.246 | 6.175 | -1036.744 | 10604.181 |

## Artifacts

- `ablation_results.csv` — per-case metrics
- `ablation_summary.csv` — aggregated means
- `degradation_ranking.csv` — ranked metric disruption per layer
- `rho_log.csv` — per-case perturbation magnitude ρ
- `intervention_strength_summary.csv` — ρ and degradation-per-ρ by layer
- `figures/heatmap_mean_degradation.png`
