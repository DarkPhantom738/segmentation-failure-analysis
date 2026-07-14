# Artifact inventory — Confidence + Consistency Triage

## Reused (no regeneration)

1. Outer folds: `outputs_consistency_failure_detection/fold_assignments.csv` (seed 42, 5×75)
2. Confidence: `outputs_layer_aware_latent_risk/case_level_confidence_features.csv`
   - GT-free TTA summaries; **375/375 exact** match vs retained `*_pred_tta.npy`
3. Morphology / labels: predicted-mask anatomy + Dice labels from consistency table
4. Layer embeddings: `outputs_10hour/layer_embeddings/` (9 layers × 375)
5. GT anatomy for probe training only via `build_anatomy_table`

## Regenerated inside nested CV

- Anatomical Ridge probes (best layer per target on outer-train only)
- Consistency gap features (train-fold scaling)
- Logistic models, feature-selection mode, Platt calibration

## Forbidden

- GT-linked uncertainty columns from `failure_metrics.csv`
