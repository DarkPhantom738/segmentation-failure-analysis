# Layer Holdout Recoverability Report

**Cases:** 375 (187 selection / 188 locked)  
**Split seed:** 42 (stratified by Dice < 0.8)  
**Probe:** Ridge regression, scaler fit per fold (selection) or on selection only (locked)

## Key targets — selection OOF R² vs locked test R²

| Target | Best layer (selection) | Sel R² | Locked R² (best) | Locked R² (bottleneck) | Δ locked |
|--------|------------------------|--------|--------------------|-------------------------|----------|
| Centroid Z | bottleneck | 0.936 | 0.926 | 0.926 | +0.000 |
| log(WT volume) | decoder1 | 0.740 | 0.773 | 0.279 | +0.494 |
| Boundary complexity | decoder2 | 0.412 | 0.302 | -0.006 | +0.307 |
| Edema fraction | decoder1 | 0.554 | 0.593 | 0.010 | +0.583 |
| Dice | decoder1 | 0.463 | 0.548 | 0.257 | +0.291 |

## All layers × targets (locked test R²)

See `locked_test_recoverability.csv` and `figures/heatmap_locked_r2.png`.

## Notes

- Selection-set R² is OOF within the selection half (used to rank layers).
- Locked-test R² is a single evaluation: train on full selection half, test on locked half.
- Layer choice per target uses selection OOF only; locked R² for the chosen layer is not peeked during selection.
