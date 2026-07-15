# Reporting notes (author-facing)

Internal cautions for manuscript wording. Not required for using the repository.

## Preferred terms

- **Predictive entropy**, **TTA predictive entropy**, or **case-level confidence features** for summaries from 8-fold TTA-averaged probabilities. Do not call these epistemic uncertainty or mutual information unless a separate MI analysis is added.
- **Matched sliding-window baseline** for ablation scoring (same patch grid/overlap as the ablation forward pass; no TTA).
- **Functional dependence on intact spatial activations** for whole-layer mean ablation. Ablation is property-agnostic and does not prove that a single probe direction is causally necessary.
- **Weak semantic steering** for decoder1 edema edits in the matched-random screen.
- **Decodable but not probe-specific downstream control** for decoder2 whole-tumor voxel-count edits.
- **Representation–output consistency / anatomical discrepancy** for gaps between representation probes and predicted-mask anatomy.
- **Failure triage** for ranking cases for review—not automatic repair.

## Number sources

- Full-cohort probe R²: `outputs_10hour/layer_analysis/layer_recoverability.csv` (375-case OOF).
- Locked-holdout heatmaps use a separate selection/test split; do not treat those R² values as interchangeable with the full-cohort table.
- Canonical failure triage snapshot: `results/paper/triage_20260712/`.
- Validation snapshot: `results/paper/method_validation/`.
- Confidence features snapshot: `results/paper/confidence_features.csv` (GT-free).

## Claims to avoid

- That editing repairs segmentations or meaningfully improves Dice.
- That consistency replaces confidence, or that it fixes edema-compartment failures.
- External generalization, clinical deployment, or automatic correction.
- Mixing matched-baseline and TTA-baseline ablation numbers as if they were the same primary endpoint.
