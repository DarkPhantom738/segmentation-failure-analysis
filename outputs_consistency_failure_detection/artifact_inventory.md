# Artifact inventory for Representation–Output Consistency Score

## Available (375 validation cases)

1. **Layer embeddings** — `outputs_10hour/layer_embeddings/` (all 9 layers × 375)
2. **Case-level OOF probe predictions** — **NOT stored**; refit inside nested CV
3. **Ground-truth anatomical targets** — via `build_anatomy_table` (probe training / labels only)
4. **Predicted-segmentation measurements** — from `path_prediction` (375/375)
5. **Dice / boundary error** — `failure_metrics.csv` (evaluation labels only)
6. **Probability / uncertainty maps** — **missing on disk** (`path_entropy` 0/375)
7. **Predicted masks** — 375/375
8. **Best layers** — selected inside each outer-training fold (not globally)
9. **Existing probe predictions OOF?** — aggregate catalog R² only; case-level preds regenerated in CV
10. **Confidence baseline** — **unavailable without leakage**. Cached columns
    (`mean_entropy_error`, `entropy_error_auroc`, `overlap_top_*`,
    `confident_false_negative_fraction`) use GT error masks in `analyze.py`.
    Inference-time confidence would require regenerating entropy/probability maps.
