# Anatomical Representation–Output Consistency Improves Confidence-Based Failure Triage in 3D Brain Tumor Segmentation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: BraTS](https://img.shields.io/badge/dataset-BraTS%202021-green.svg)](https://www.synapse.org/#!Synapse:syn27046444)

This project studies the internal representations of a 3D U-Net for BraTS brain-tumor segmentation. Anatomical properties can often be decoded from those activations, but probe-derived edit directions did not consistently control or repair the final mask. What did help was measuring when the anatomy implied by a representation disagrees with the anatomy of the predicted mask, then combining those disagreements with ordinary inference-time confidence.

In one sentence: this project helps a brain-tumor segmentation model flag which of its predictions are most likely to need human review.

---

## Main result

On 375 BraTS 2021 validation cases, nested cross-validation (5 outer / 4 inner folds, seed 42) with 5000 paired bootstrap replicates.

**Detecting the lowest-quality 20% of segmentations** (by mean foreground Dice):

| Method | AUPRC | Capture at 20% review |
|---|---:|---:|
| Confidence only | 0.805 | 71.4% |
| Confidence + morphology | 0.851 | 75.3% |
| Confidence + pooled representations | 0.854 | 74.0% |
| Confidence + representation–output consistency | **0.895** | **79.2%** |

- AUPRC gain vs confidence: **+0.090** (paired bootstrap 95% CI **[0.031, 0.152]**).
- Confidence + consistency beat confidence in about **99.8%** of bootstrap samples.
- For mean foreground Dice &lt; 0.70, AUPRC improved from **0.770** to **0.887**.
- Benefit for edema-specific failure labels was limited (morphology-augmented confidence often did better there).

These are research quality-control metrics—not a claim of clinical benefit or deployment readiness.

Canonical numbers: `outputs_confidence_consistency_triage_20260712_030902/aggregate_metrics.csv` and `bootstrap_comparisons.csv`. Broader baseline tables: `outputs_method_validation/`.

---

## What representation–output consistency means

For each anatomical property (for example enhancing-tumor fraction, edema fraction, or whole-tumor volume):

1. Fit a Ridge probe on a hidden layer (using ground truth only inside the training fold).
2. Read the same property from the **predicted** segmentation mask.
3. Record how far the two disagree.
4. Combine those gaps with confidence features (TTA max-probability, entropy, margin summaries, and related case-level statistics).
5. Score case-level failure risk with a logistic model under nested CV.

Gap features (conceptual; code column names may differ):

```text
signed_gap   = representation_estimate - predicted_mask_measurement
absolute_gap = abs(representation_estimate - predicted_mask_measurement)
relative_gap = abs(representation_estimate - predicted_mask_measurement)
               / (abs(predicted_mask_measurement) + epsilon)
```

**Inference-time rule:** the triage score does not take ground-truth masks or GT-linked error maps as inputs. Ground truth is used only to fit probes inside training folds, define failure labels, and evaluate held-out performance.

Refer to features as representation–output enhancing-fraction gaps, edema-fraction gaps, whole-tumor-volume gaps, and so on—even when code columns still use historical names.

---

## Study overview

| Item | Setting |
|---|---|
| Dataset | BraTS 2021 |
| Train / analysis split | 876 train / **375** internal validation |
| Inputs | T1, T1ce, T2, FLAIR |
| Model | Four-class 3D U-Net |
| Checkpoint | Epoch 5 from the ~10-hour training config (`configs/ten_hour.yaml`) |
| Stages analyzed | encoder1–4, bottleneck, decoder4–1 (nine stages) |

Four connected analyses on the same frozen model:

1. **Layer-wise linear probing** — full-cohort out-of-fold Ridge recoverability.
2. **Spatial mean ablation** — replace each layer map with its channel means; score Dice drop vs a matched sliding-window baseline.
3. **Probe-aligned editing** — add scaled probe directions; compare to matched random directions (exploratory **30-case** screens, plus full-cohort edit summaries).
4. **Confidence-augmented failure triage** — consistency gaps + confidence under leakage-safe nested CV (**375** cases).

Train/val provenance: `docs/manuscript/environment_and_split.md`.

---

## Mechanistic companion findings

Decode ≠ control is still in the paper, but it is no longer the headline endpoint.

### Full-cohort probe R² (375 cases, 5-fold OOF)

Source: `outputs_10hour/layer_analysis/layer_recoverability.csv`

| Property | Best layer | R² |
|---|---|---:|
| Whole-tumor volume (voxel count) | decoder2 | 0.822 |
| Enhancing-tumor fraction | decoder1 | 0.647 |
| Edema fraction | decoder1 | 0.596 |
| Dice | decoder1 | 0.565 |
| Necrotic / nonenhancing fraction | decoder1 | 0.525 |
| Boundary complexity | decoder2 | 0.430 |
| Boundary error | decoder1 | 0.415 |

The locked-holdout heatmap below uses a **separate** selection/test split, so its R² values are **not** the same estimates as this table.

### Editing (screens + full cohort)

Matched-random screens (n = 30 each; exploratory, not powered tests):

- Edema @ decoder1: probe / random absolute response ratio **2.36** (weak probe-specific steering).
- Whole-tumor volume @ decoder2: probe / random ratio **0.86**—strongly decodable, but the probe direction did not beat random for changing measured volume. Analytical Ridge movement (+α/−α opposite sign) was far more consistent than actual segmentation volume flips.

Full-cohort edema edits were small and monotonic; Dice barely moved. Editing is not a repair method here.

### Mean ablation (375 cases, matched baseline)

Dice degradation under spatial mean replacement:

| Layer | Mean Dice degradation |
|---|---:|
| decoder1 | 0.889 |
| encoder1 | 0.498 |
| decoder2 | 0.399 |
| bottleneck | ≈ 0 |

Spatial organization at the bottleneck was relatively insensitive to mean replacement under this intervention. That does **not** mean the bottleneck is unnecessary for the network as a whole.

Source: `outputs_10hour/layer_interventions/matched_baseline/baseline_comparison.csv`.

---

## Why this matters

Internal maps can encode anatomy. That does not automatically give a handle for reliable control or automatic correction. The same encoding can still be useful for monitoring: when the representation’s anatomical story disagrees with the mask the network emits, that mismatch adds information beyond confidence for spotting **overall** bad cases that may need review.

---

## Figures

**1. Layer recoverability (locked holdout)** — different split than the full-cohort table above.

![Locked-holdout recoverability heatmap](outputs_10hour/layer_holdout_recoverability/figures/heatmap_locked_r2.png)

**2. Whole-tumor volume: analytical probe response vs actual mask volume** (30-case screen).

![Analytical vs actual volume](outputs_10hour/volume_probe_screen/figures/analytical_vs_actual_volume.png)

**3. Failure-triage precision–recall** (canonical nested run).

![Precision–recall curves](outputs_confidence_consistency_triage_20260712_030902/figures/precision_recall_curves.png)

**4. Failure capture vs review budget.**

![Failure capture curves](outputs_method_validation/figures/failure_capture_curves.png)

---

## Repository structure

```text
configs/
  ten_hour.yaml
  confidence_consistency_triage.yaml
  method_validation.yaml
  consistency_failure_detection.yaml
  layer_aware_latent_risk.yaml
src/
  data/                 # BraTS loading / preprocessing
  models/               # 3D U-Net (+ exploratory repair modules)
  analysis/             # probing, editing, consistency, triage
outputs_10hour/
  layer_analysis/
  layer_holdout_recoverability/
  layer_interventions/
  representation_editing/
  edema_probe_screen/
  volume_probe_screen/
  failure_tables/
outputs_confidence_consistency_triage/
outputs_confidence_consistency_triage_20260712_030902/   # preferred canonical triage
outputs_method_validation/
outputs_layer_aware_latent_risk/                         # confidence CSV
outputs_consistency_failure_detection/
docs/
  manuscript/environment_and_split.md
  reporting_notes.md
```

Large artifacts (raw BraTS volumes, checkpoints, probability maps, layer `.npy` embeddings) are **not** stored in Git.

---

## Reproduction

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Point data.root at your local BraTS tree in configs/ten_hour.yaml
```

### 2. Data and checkpoint

You need locally:

- BraTS 2021 training data;
- epoch-5 / latest checkpoint under `outputs_10hour/checkpoints/` (not in Git);
- `outputs_10hour/failure_tables/failure_metrics.csv` (committed) for case lists and paths.

### 3. Representation extraction and probing

```bash
python export_layer_embeddings.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --output-dir outputs_10hour/layer_embeddings

python analyze_layer_holdout_recoverability.py \
  --layer-index outputs_10hour/layer_embeddings/layer_embedding_index.csv \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv

python learn_semantic_directions.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --layer-index outputs_10hour/layer_embeddings/layer_embedding_index.csv \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --output-dir outputs_10hour/semantic_directions
```

### 4. Ablation and editing

```bash
python analyze_representation_editing.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --directions-dir outputs_10hour/semantic_directions \
  --output-dir outputs_10hour/representation_editing

python analyze_edema_probe_screen.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv

python analyze_volume_probe_screen.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --n-random 5

python analyze_layer_interventions.py \
  --config configs/ten_hour.yaml \
  --data-root /path/to/BraTS2021_Training_Data \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --output-dir outputs_10hour/layer_interventions

python analyze_layer_interventions.py \
  --config configs/ten_hour.yaml \
  --data-root /path/to/BraTS2021_Training_Data \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --output-dir outputs_10hour/layer_interventions \
  --recompute-matched-baseline
```

### 5. Confidence-feature generation

Committed confidence table: `outputs_layer_aware_latent_risk/case_level_confidence_features.csv`.

To regenerate (needs checkpoint + local volumes; GPU/CPU heavy):

```bash
python scripts/run_layer_aware_latent_risk.py \
  --config configs/layer_aware_latent_risk.yaml \
  --stage confidence
```

### 6. Final confidence + consistency evaluation

Committed results can be inspected without re-running:

```bash
ls outputs_confidence_consistency_triage_20260712_030902
cat outputs_method_validation/validation_summary.md
```

To recompute (needs layer embeddings + feature tables; may write a timestamped sibling directory):

```bash
python scripts/run_consistency_failure_detection.py \
  --config configs/consistency_failure_detection.yaml

python scripts/run_confidence_consistency_triage.py \
  --config configs/confidence_consistency_triage.yaml

python scripts/run_method_validation.py \
  --config configs/method_validation.yaml
```

---

## Limitations and intended use

- One public dataset (BraTS 2021) and one U-Net in the main analysis.
- No external cohort; retrospective computational study only.
- Limited benefit for edema-specific failure definitions.
- Enhancing-fraction discrepancy features carry much of the consistency signal.
- No supported improvement in risk–coverage AURC.
- Representation editing did not provide automatic correction.
- Exploratory edit screens use 30 cases; triage claims rest on the 375-case nested evaluation.



---

## Data governance

Code is released for research. BraTS 2021 use follows [Synapse terms](https://www.synapse.org/#!Synapse:syn27046444). Committed tables do not include patient identifiers. BraTS references: Menze et al., CVPR 2015; Bakas et al., 2017–2021.

No `LICENSE` file is present yet; license terms have not been finalized.
