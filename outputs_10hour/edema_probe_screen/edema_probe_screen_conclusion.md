# Edema Probe Screening Conclusion

**The probe direction clearly outperforms matched random perturbations** on mean |Δ edema fraction| at |α|=1 with RMS-matched unit-norm directions.

| Metric | Probe | Random (mean) | Ratio |
|--------|------:|--------------:|------:|
| mean \|Δ edema fraction\| | 0.02079 | 0.00880 | 2.36 |

- Probe edema effects loaded from `outputs_10hour/representation_editing/editing_results.csv` (|α|=1).
- Random directions: 3 fixed unit vectors, same |α| magnitude as probe, decoder1→head only from cached activations.
- Analytical probe-prediction Δ uses pooled GAP + scaler + Ridge coef/intercept (no segmentation).

**Interpretation:** A ratio near 1 means the edema shift is generic to any same-sized perturbation; a ratio >1.25 suggests the learned direction is somewhat specific.
