# Causal Representation Editing Report

Validation cases: **375**

## Method

Semantic directions are learned by fold-safe Ridge probes on pooled layer
readouts, then lifted into activation tensor space via the adjoint of global
average pooling (spatially constant per-channel perturbation). Edits:
`A' = A + α·Δ` on the activation map consumed by downstream computation.

**Encoder layers:** edits target the **skip path** at decode fusion (the same
tensor GAP'd for probing), not activations propagated into deeper encoders.

**Baseline:** alpha=0 through the same sliding-window editing inference path
(not cached failure-table predictions).

**Selectivity at |α|=1:** directional effect = (Δ(+1) − Δ(−1)) / 2.

## Selectivity (|α| = 1)

| Direction | Selectivity | Off-target leakage |
|-----------|-------------|-------------------|
| gt_necrosis_frac (decoder1) | 0.157 | 0.378 |
| gt_edema_frac (decoder1) | 0.140 | 0.674 |
| gt_enhancing_frac (decoder1) | 0.139 | 0.647 |
| boundary_error_fraction (decoder1) | 0.107 | 0.383 |
| gt_wt_voxels (decoder2) | 0.059 | 0.007 |
| dice (decoder1) | 0.049 | 0.255 |
| gt_compactness (decoder2) | 0.043 | 0.003 |

## Monotonicity (on-target)

- **boundary_error_fraction** (decoder1): ρ=1.000, p=0.0000
- **dice** (decoder1): ρ=1.000, p=0.0000
- **gt_edema_frac** (decoder1): ρ=-1.000, p=0.0000
- **gt_enhancing_frac** (decoder1): ρ=-1.000, p=0.0000
- **gt_necrosis_frac** (decoder1): ρ=-1.000, p=0.0000
- **gt_compactness** (decoder2): ρ=-0.143, p=0.7872
- **gt_wt_voxels** (decoder2): ρ=-1.000, p=0.0000

## Interpretation

High selectivity means editing one semantic direction primarily moves its
intended output property. Off-target effects quantify disentanglement limits.
Compare with ablation results: probes may be recoverable without causal
necessity (bottleneck location), while edits test fine-grained steerability.