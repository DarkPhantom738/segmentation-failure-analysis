# Archived: Uncertainty and Latent Geometry for Segmentation Quality Estimation

This document preserves the **earlier research direction** that preceded the mechanistic probing-and-editing study now described in the main [README](../../README.md). It is kept for context and reproducibility of the 533-case uncertainty baseline experiments.

---

## Motivation

MRI brain tumor segmentation models routinely report high average Dice scores on benchmark datasets. In clinical use, however, performance is judged **case by case**: a single poor segmentation on a difficult scan can have meaningful consequences, and the clinician does not have access to the ground-truth segmentation at inference time.

A natural approach is to estimate whether a segmentation can be trusted using **predictive uncertainty**—for example, entropy of the softmax distribution under test-time augmentation (TTA). Uncertainty-based quality estimation is an active and well-studied direction.

This phase of the project asked a related but distinct question: **does the model's own bottleneck latent representation encode information about segmentation quality that is not already captured by deployable uncertainty summaries?**

---

## Research questions (archived)

1. Can latent bottleneck representations predict segmentation failures (e.g., cases with Dice below a clinical threshold)?
2. Do latent representations encode different information than uncertainty?
3. What information is stored in the latent representation—anatomy, tumor burden, failure morphology, distributional novelty?
4. When does combining latent representations with uncertainty improve failure detection?
5. Which cases does the combined model rescue that uncertainty alone misses?

---

## Pipeline (533-case primary run)

| Step | Script | Description |
|------|--------|-------------|
| 1 | `train.py` | Train 3D U-Net on BraTS |
| 2 | `train.py --export-only` | Export predictions, softmax, bottleneck embeddings |
| 3 | `train.py --export-uncertainty` | TTA-averaged probabilities and entropy |
| 4 | `analyze_failures.py` | Failure taxonomy metrics |
| 5 | `analyze_geometry.py` | UMAP, kNN label agreement |
| 6 | `compare_baselines.py` | CV: uncertainty vs geometry vs combined |
| 7 | `analyze_embedding_signal.py` | Mechanistic embedding analysis |
| 8 | `analyze_rescued_bad_cases.py` | Rescued failure case study |

Config: `configs/five_epoch_533.yaml` (400 train / 133 validation, 5 epochs).

---

## Key results (533 validation cases)

### Bad-case detection (Dice < 0.80, logistic regression, 5-fold CV)

| Feature set | AUROC | AUPRC | F1 |
|-------------|-------|-------|-----|
| Uncertainty only | **0.886** | 0.641 | 0.533 |
| Geometry only | 0.764 | 0.457 | 0.368 |
| Combined | **0.909** | 0.633 | 0.452 |

### Continuous Dice regression (Ridge CV)

| Feature set | Pearson *r* | R² |
|-------------|-------------|-----|
| Uncertainty only | **0.60** | **0.36** |
| Geometry only | 0.41 | 0.15 |
| Combined | 0.43 | 0.16 |

### Summary observations

- Uncertainty was the strongest standalone quality signal.
- Combining embeddings with uncertainty modestly improved severe-failure detection (AUROC +0.023).
- Embeddings correlated with morphology (boundary error, tumor volume) more than with entropy alone.
- A small subset of bad cases (2/17) were missed by uncertainty but flagged by the combined model—boundary-heavy, low-entropy failures. Treat as hypothesis-generating (*n* = 2).

---

## Figures

Figures from the 533-case run are in [`docs/figures/`](../figures/):

- `umap_failure_labels.png` — UMAP colored by failure label
- `umap_colored_by_dice.png` — UMAP colored by Dice
- `scatter_uncertainty_vs_dice.png` — Uncertainty vs Dice
- `calibration_comparison.png` — Calibration: uncertainty vs combined
- `rescued_bad_cases_morphology_comparison.png` — Group A vs Group B morphology
- `scatter_nn_distance_vs_dice.png` — NN distance vs Dice

---

## Reproduction (533-case run)

```bash
python train.py --config configs/five_epoch_533.yaml --epochs 5
python train.py --config configs/five_epoch_533.yaml \
  --export-uncertainty \
  --checkpoint outputs_5epoch_533/checkpoints/checkpoint_epoch_005.pt
python analyze_failures.py \
  --metrics outputs_5epoch_533/metrics_uncertainty.csv \
  --output outputs_5epoch_533/failure_tables/failure_metrics.csv
python analyze_geometry.py \
  --failure-table outputs_5epoch_533/failure_tables/failure_metrics.csv \
  --output-dir outputs_5epoch_533/geometry
python compare_baselines.py \
  --failure-table outputs_5epoch_533/failure_tables/failure_metrics.csv \
  --geometry-table outputs_5epoch_533/geometry/umap_coordinates.csv \
  --output-dir outputs_5epoch_533/baselines \
  --bad-case-mode threshold --dice-threshold 0.80
```

Or: `bash scripts/run_midscale_pipeline.sh`

---

## Relation to current work

The failure-detection experiments showed that internal representations carry morphological structure beyond deployable uncertainty. That observation motivated the current study: **if representations encode anatomy, can we read it out (probe) and manipulate it (edit)?** The main README documents that follow-on mechanistic analysis on the 375-case 10-hour U-Net run.

---

## Limitations (archived phase)

- Single dataset (BraTS 2021), single architecture.
- Modest combined-model gains (+0.02–0.03 AUROC).
- Rescued-case subgroup analyses underpowered.
- No prospective clinical validation.
