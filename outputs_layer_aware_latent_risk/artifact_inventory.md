# Artifact inventory — Layer-Aware Latent Risk Triage

Generated for Stage 1 verification. Do not treat prior consistency “confidence” columns as usable.

## 1. Consistency failure-detection outputs

Directory: `outputs_consistency_failure_detection/`

Present:
- `fold_assignments.csv` — 375 cases × 5 outer folds; exactly one `test` row per case
- `case_level_features.csv` — morphology, held-out probe/gap features, quality labels
- `outer_fold_predictions.csv` — one held-out score/prediction row per case
- metrics / bootstrap / figures / `feasibility_report.md`

## 2. Outer-fold assignments

- Source: `outputs_consistency_failure_detection/fold_assignments.csv`
- Design: `KFold(n_splits=5, shuffle=True, random_state=42)` over 375 cases
- Split sizes: 75 test / 300 train per outer fold
- **Reuse for this experiment** (identical folds across all methods)

## 3–5. Layer embeddings (pooled representations)

Index: `outputs_10hour/layer_embeddings/layer_embedding_index.csv` (375/375)

| Layer | Dim | Present |
|---|---:|---|
| encoder1 | 32 | 375 |
| encoder2 | 64 | 375 |
| encoder3 | 128 | 375 |
| encoder4 | 256 | 375 |
| bottleneck | 128 | 375 |
| decoder4 | 256 | 375 |
| decoder3 | 128 | 375 |
| decoder2 | 64 | 375 |
| decoder1 | 32 | 375 |

Concatenated all-layer dim = 1088. These are **globally pooled** vectors (not spatial maps).

## 6. Morphology features (pred-mask only)

In `case_level_features.csv`:
- `pred_tumor_volume`, `pred_edema_volume`, `pred_enhancing_volume`, `pred_necrosis_volume`
- `out_log_wt_volume`, `out_gt_edema_frac`, `out_gt_enhancing_frac`, `out_gt_necrosis_frac`
- `out_gt_compactness`, `n_components`, `surface_to_volume`

## 7. Dice / quality labels

- `label_wt_dice`, `label_mean_fg_dice`, `label_edema_dice`, `label_boundary_error`
- Also in `outer_fold_predictions.csv` / `failure_metrics.csv`

## 8. Inconsistency features

Held-out signed/absolute/standardized/relative gaps for:
`log_wt_volume`, `gt_edema_frac`, `gt_enhancing_frac`, `gt_necrosis_frac`, `gt_compactness`

Plus held-out `rep_dice`, `rep_boundary_error_fraction` (direct quality probes).

## 9. Softmax probability maps

**Not cached.** No `*softmax*` / probability volumes under `outputs_10hour/`.
Only hard TTA pred masks exist (`predictions/epoch_005/*_pred_tta.npy`).

## 10. Case-level confidence summaries

**No inference-time confidence summaries exist.**

Cached `failure_metrics.csv` columns (`mean_entropy_error`, `overlap_top_*`,
`confident_false_negative_fraction`, …) use **GT error masks** and must not be reused.

→ Must regenerate GT-free confidence via frozen sliding-window inference
  (epoch-5 checkpoint), saving **case-level summaries only**.

## 11. One outer-fold held-out prediction per case?

**Yes** for the consistency experiment (`outer_fold_predictions.csv`, 375 rows;
fold test membership unique).

## 12. Prior pooled-representation leakage status

Previous `pooled_repr` method:
- Concatenated **all 9 layers** (dim 1088)
- Nested CV: scaler + logistic C tuned on outer-train only
- **Did not** select layers or PCA on all 375 cases
- **Did not** compare single-layer / fusion / PCA variants inside folds

So the prior 0.811 AUPRC is nested-CV held-out for concat-all, but **not** a
validated layer-aware selection result. This experiment must re-evaluate with
strict per-fold layer/PCA/fusion selection.

## Checkpoint / data for confidence regen

- Checkpoint: `outputs_10hour/checkpoints/checkpoint_epoch_005.pt`
- Data root: `data/BraTS2021_Training`
- Cache: `outputs_10hour/cache`
- Config template: `configs/ten_hour.yaml` (override `data.root`)

## Missing before full nested evaluation

1. GT-free case-level confidence features (requires one frozen inference pass)
2. Layer-aware nested-CV predictions (scaffolded; `stage=full` gated)
3. Bootstrap (5000), calibration, robustness (seeds / permutation)

## How to proceed (do not run full yet)

```bash
# Stage 1 (done)
python scripts/run_layer_aware_latent_risk.py --stage inventory
python scripts/run_layer_aware_latent_risk.py --stage dry_check

# Stage 2 when ready (long; frozen epoch-5 inference)
python scripts/run_layer_aware_latent_risk.py --stage confidence

# Smoke test only
python scripts/run_layer_aware_latent_risk.py --stage confidence --max-cases 2

# Full nested CV — not enabled yet
# python scripts/run_layer_aware_latent_risk.py --stage full
```

## Runtime verification

- Cases aligned: 375
- Confidence CSV present: True (375/375)
- Exact TTA mask match vs retained `*_pred_tta.npy`: **375/375**
- See `confidence_consistency_report.md`

