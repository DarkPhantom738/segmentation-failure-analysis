# Layer Analysis Report

**Validation cases:** 375

## Research questions

### Does the bottleneck discard fine morphology?

Compare bottleneck R² for boundary complexity (0.009) vs best layer (Dec 2: 0.430).
If decoder/early-encoder layers exceed the bottleneck on boundary, composition, or failure-morphology targets, the bottleneck compresses fine detail.

### Which layer best encodes tumor location?

Best layer for centroid Z: **Bottleneck** (R² = 0.940).

### Which layer best encodes tumor volume?

Best layer for log(WT volume): **Dec 2** (R² = 0.752).

### Which layer best encodes boundary complexity?

Best layer: **Dec 2** (R² = 0.430); bottleneck R² = 0.009.

### Which layer best encodes tissue composition?

| Subregion fraction | Best layer | R² |
|--------------------|------------|-----|
| Edema | Dec 1 | 0.596 |
| Enhancing | Dec 1 | 0.647 |
| Necrosis | Dec 1 | 0.525 |

### Which layer best predicts segmentation failure (Dice)?

Best layer for Dice regression: **Dec 1** (R² = 0.565).

### Which layer best predicts bad cases (Dice < 0.80)?

| Method | AUROC |
|--------|-------|
| Uncertainty only | 0.929 |
| Best layer embedding (Dec 1) | 0.921 |
| Uncertainty + best layer | 0.970 |

### Do skip/decoder features contain information missing from the bottleneck?

See `layer_recoverability.csv` heatmap (`figures/heatmap_layer_target_r2.png`).
Targets where decoder or encoder layers beat bottleneck R² indicate distributed anatomical coding.

### Does any layer improve over uncertainty for quality estimation?

Primary hypothesis is **not** that embeddings beat uncertainty. On this run, uncertainty AUROC = 0.929; best layer = 0.921; combined = 0.970.

## Artifacts

- `layer_recoverability.csv`
- `layer_quality_results.csv`
- `figures/heatmap_layer_target_r2.png`
- `figures/line_recoverability_across_depth.png`
- `figures/bar_badcase_auroc_by_layer.png`
- `figures/bar_uncertainty_vs_best_layer.png`
