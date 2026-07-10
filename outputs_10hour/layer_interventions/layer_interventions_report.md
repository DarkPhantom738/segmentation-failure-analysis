# Layer Intervention Report (Part 1 — Ablation)

**Cases:** 375  
**Interventions:** zero, mean, noise (per-layer activation replacement)  
**Baseline:** TTA validation predictions from `failure_metrics.csv` (**robustness analysis**; primary results use matched sliding-window baseline in `matched_baseline/`)

## Ablation semantics

| Layer | Intervention type |
|-------|-------------------|
| encoder1–encoder4 | **Skip ablation** — encoder runs normally; only the skip tensor at decode fusion is replaced |
| bottleneck | Replace bottleneck map, then decode |
| decoder4–decoder1 | Replace decoder block output, continue downstream decoding |

**Interpretation:** mean ablation is the primary read (removes case-specific spatial detail while preserving scale). Zero and noise are sanity checks. Compare layers using `intervention_strength_summary.csv` (degradation per unit ρ) — raw degradation alone is not comparable across layers. Results measure **functional dependence on intact spatial activations**, not necessity of any single probe direction.

Positive `mean_degradation` means the metric got worse after ablation.

**Note:** Prefer `matched_baseline/` for manuscript tables. Relative to that matched baseline, TTA-baseline Dice degradations differ by a uniform −0.006 and preserve the same layer ranking (Spearman ρ = 1.000; see `matched_baseline/baseline_comparison.md`).

## Perturbation strength (ρ = ||A - A'|| / ||A||)

| Layer | zero ρ | mean ρ | noise ρ | Dice deg/ρ (mean) |
|-------|--------|--------|---------|-------------------|
| encoder1 | 1.000 | 0.806 | 1.284 | 0.612 |
| encoder2 | 1.000 | 0.829 | 1.299 | 0.322 |
| encoder3 | 1.000 | 0.910 | 1.352 | 0.024 |
| encoder4 | 1.000 | 0.917 | 1.358 | -0.006 |
| bottleneck | 1.000 | 0.859 | 1.323 | -0.007 |
| decoder4 | 1.000 | 0.689 | 1.215 | -0.011 |
| decoder3 | 1.000 | 0.849 | 1.312 | 0.059 |
| decoder2 | 1.000 | 0.839 | 1.305 | 0.469 |
| decoder1 | 1.000 | 0.805 | 1.284 | 1.098 |

## Top degradations — zero ablation


### encoder1

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 7924.6080 |
| 2 | False-negative voxels | 7270.4640 |
| 3 | Boundary complexity | 1.8735 |

### encoder2

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 14042.9600 |
| 2 | False-positive voxels | 13493.0000 |
| 3 | Boundary complexity | 6.9665 |

### encoder3

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 758.5280 |
| 2 | False-positive voxels | 360.7387 |
| 3 | False-negative voxels | 18.4027 |

### encoder4

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 280.0187 |
| 2 | False-negative voxels | 59.7973 |
| 3 | Boundary complexity | 0.1639 |

### bottleneck

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 287.2827 |
| 2 | False-negative voxels | 5.1440 |
| 3 | Boundary complexity | 0.1762 |

### decoder4

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 331.6453 |
| 2 | False-negative voxels | 70.2827 |
| 3 | Boundary complexity | 0.1882 |

### decoder3

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 2460.6800 |
| 2 | False-negative voxels | 1693.6907 |
| 3 | Boundary complexity | 0.5761 |

### decoder2

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 22482.1573 |
| 2 | False-positive voxels | 21918.2027 |
| 3 | Boundary complexity | 9.4342 |

### decoder1

| Rank | Metric | Mean degradation |
|------|--------|------------------|
| 1 | Tumor volume (pred WT voxels) | 11701.1067 |
| 2 | False-negative voxels | 10606.3947 |
| 3 | Boundary complexity | 6.2129 |

## Cross-layer comparison (zero ablation, mean degradation)

| Layer | Dice | Volume | Edema frac | Enh frac | Boundary | FP | FN |
|-------|------|--------|------------|----------|----------|----|----|
| encoder1 | 0.491 | 7924.608 | 0.519 | 0.149 | 1.874 | -565.888 | 7270.464 |
| encoder2 | 0.341 | 14042.960 | 0.140 | 0.085 | 6.967 | 13493.000 | -177.347 |
| encoder3 | 0.019 | 758.528 | 0.031 | 0.014 | 0.295 | 360.739 | 18.403 |
| encoder4 | -0.006 | 280.019 | 0.013 | 0.008 | 0.164 | -121.240 | 59.797 |
| bottleneck | -0.006 | 287.283 | 0.013 | 0.007 | 0.176 | -60.229 | 5.144 |
| decoder4 | -0.007 | 331.645 | 0.014 | 0.008 | 0.188 | -155.261 | 70.283 |
| decoder3 | 0.048 | 2460.680 | 0.046 | 0.029 | 0.576 | -755.416 | 1693.691 |
| decoder2 | 0.406 | 22482.157 | 0.159 | 0.098 | 9.434 | 21918.203 | -563.955 |
| decoder1 | 0.883 | 11701.107 | 0.624 | 0.245 | 6.213 | -1094.712 | 10606.395 |

## Artifacts

- `ablation_results.csv` — per-case metrics
- `ablation_summary.csv` — aggregated means
- `degradation_ranking.csv` — ranked metric disruption per layer
- `rho_log.csv` — per-case perturbation magnitude ρ
- `intervention_strength_summary.csv` — ρ and degradation-per-ρ by layer
- `figures/heatmap_mean_degradation.png`
