# Decodable Is Not Necessarily Controllable  
## Probing and Editing Anatomical Representations in a 3D U-Net

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: BraTS](https://img.shields.io/badge/dataset-BraTS%202021-green.svg)](https://www.synapse.org/#!Synapse:syn27046444)

### Project summary

This project investigates what anatomical and segmentation-related information is represented inside a trained 3D U-Net for brain-tumor segmentation, whether that information is required for prediction, and whether it can be causally manipulated through representation editing.

The central finding is that anatomical information can be highly linearly decodable without providing an effective causal control direction.

In particular, tumor volume is strongly recoverable from decoder2 activations with an out-of-fold Ridge-probe R² of 0.822. However, editing activations along that probe direction changes the probe prediction in the expected direction while producing only negligible, non-specific changes in the final segmentation volume.

By contrast, decoder1 edema directions permit weak but measurable semantic steering, although the resulting changes are small and affect other tissue properties.

---

## Research question

**Does the presence of linearly decodable anatomical information in a segmentation network imply that the same representation can be selectively controlled?**

The project separates three forms of evidence:

1. **Recoverability:** Can a property be decoded from internal activations?
2. **Functional dependence:** Does disrupting the layer damage segmentation?
3. **Controllability:** Can a targeted intervention selectively change the corresponding output property?

These forms of evidence are often treated as interchangeable in representation-analysis studies. This project tests where they agree and where they diverge.

---

## Main findings

The analysis uses 375 validation cases and a trained 3D U-Net checkpoint.

| Property | Layer | Probe performance | Editing result |
|---|---|---:|---|
| Tumor volume | decoder2 | R² = 0.822 | No probe-specific downstream control |
| Enhancing fraction | decoder1 | R² = 0.647 | Weak, monotonic steering |
| Edema fraction | decoder1 | R² = 0.596 | Weak, monotonic steering |
| Dice | decoder1 | R² = 0.565 | Negligible editing effect |
| Necrosis fraction | decoder1 | R² = 0.525 | Weak, coupled steering |
| Boundary complexity | decoder2 | R² = 0.430 | No meaningful editing effect |

### Edema editing

At decoder1, editing along the learned edema probe direction produced:

- approximately 2.0 percentage points of edema-fraction change at |α| = 1;
- approximately 4.1 percentage points at |α| = 2;
- a monotonic dose response across tested intervention strengths;
- a mean absolute effect 2.36 times larger than RMS-matched random directions in a 30-case screening experiment.

However, the edit also changed enhancing fraction, necrosis fraction, volume, and other outputs. The intervention is therefore weakly semantic but not independently selective.

### Volume probe–edit gap

Tumor volume provides the strongest example of a gap between recoverability and controllability.

- Decoder2 volume probe: R² = 0.822
- Mean downstream volume change from the probe edit: 3.82 voxels
- Mean change from RMS-matched random directions: 4.45 voxels
- Probe/random effect ratio: 0.86
- Probe direction percentile relative to random controls: 47th percentile
- Analytical probe prediction moved in opposite expected directions for +α and −α in 100% of cases
- Actual segmentation volume moved in opposite directions in only 57% of cases

The intervention therefore moves the activation along the Ridge probe’s readout axis, but that axis does not function as a specific downstream control direction for tumor volume.

---

## Current interpretation

The results support the following conclusion:

> Late U-Net decoder representations contain linearly accessible anatomical information, but linear recoverability does not guarantee selective causal control. Probe-aligned editing can weakly steer some tissue-composition properties, while highly predictive volume directions fail to produce probe-specific changes in the final segmentation.

This is not currently presented as a clinical correction method. Dice improvements are negligible, and the interventions are not sufficiently selective for automatic segmentation repair.

The intended contribution is a mechanistic analysis of the distinction between representation readout and representation control.

---

## Methods overview

### Model

- 3D U-Net for multi-class brain-tumor segmentation
- Primary analysis checkpoint: 10-hour training run, epoch 5
- Analysis performed on 375 validation cases

### Stage 1: Linear probing

Internal activations were extracted from the bottleneck and decoder layers.

Global pooled activation features were used to train fold-safe Ridge probes for:

- tumor volume;
- enhancing-tumor fraction;
- edema fraction;
- necrosis fraction;
- Dice;
- boundary complexity;
- boundary error.

Performance was evaluated using out-of-fold predictions.

### Stage 2: Layer ablation

Layer activations were replaced with mean activations to measure how strongly segmentation depended on intact layer representations.

These experiments establish functional dependence on intact decoder activations, but they are not interpreted as property-specific causal ablations.

### Stage 3: Representation editing

Activations were modified using:

A' = A + αΔ

where Δ is a direction derived from a trained linear probe.

The analysis measured:

- target-property change;
- off-target changes;
- Dice change;
- dose-response behavior;
- effects in easy and difficult cases;
- comparison with RMS-matched random directions.

---

## Why this may be scientifically useful

The project provides evidence that three commonly conflated concepts are distinct:

- a property being statistically recoverable from an activation;
- the network depending on that activation for segmentation;
- the same property being selectively steerable through intervention.

The strongest probe in the study produced one of the weakest edits. This suggests that predictive probe directions may reflect naturally occurring correlations in representation space without corresponding to directions that the downstream network uses as causal control axes.

This distinction may be relevant to:

- mechanistic interpretability in medical imaging;
- validation of representation-editing methods;
- causal analysis of segmentation networks;
- failure diagnosis and uncertainty analysis;
- the design of more appropriate intervention directions.

---

## Current limitations

The current study has several important limitations:

- The primary findings come from one trained U-Net checkpoint.
- The random-direction control experiments use a limited subset of cases and directions.
- Whole-layer mean ablation is highly destructive and does not isolate individual semantic properties.
- Tissue fractions are mathematically coupled because they are compositional variables.
- The editing intervention uses a globally broadcast additive channel direction.
- The analysis does not demonstrate clinically useful segmentation correction.
- The current results should be interpreted as a mechanistic case study rather than a universal claim across segmentation architectures.

These limitations will be stated explicitly in the manuscript.

---

## Current project status

Completed:

- layer-wise linear probing;
- whole-layer ablation analysis;
- representation editing across multiple anatomical properties;
- dose-response analysis;
- tissue-coupling analysis;
- easy-versus-failure-case analysis;
- RMS-matched random-direction screen for decoder1 edema;
- RMS-matched random-direction screen for decoder2 volume.

Current stage:

- locking the experimental results;
- constructing manuscript figures;
- writing the Results and Methods sections;
- refining the interpretation and statistical reporting.

---

## Mentorship I am seeking

I am looking for academic mentorship in one or more of the following areas:

- positioning the contribution for a medical-imaging or machine-learning audience;
- evaluating whether the central claim is sufficiently novel and well supported;
- strengthening the causal and statistical interpretation;
- identifying essential analyses versus unnecessary expansion;
- selecting an appropriate venue;
- reviewing the manuscript structure and major figures;
- avoiding overclaiming while preserving the significance of the result.

The immediate goal is not to request extensive implementation support. I am primarily seeking scientific guidance on framing, rigor, and publication strategy.

---

## Questions on which feedback would be especially valuable

1. Is the distinction between decodability and controllability a sufficiently clear central contribution?
2. Are the current random-direction controls adequate for the scope of the claim?
3. Should the paper be framed as a focused mechanistic case study of one U-Net, or is replication across additional training seeds essential before submission?
4. Are there any major causal-interpretation issues that would prevent publication in its current form?
5. Which analyses belong in the main paper versus the supplement?
6. Which journals or conferences would be the best fit?

---

## Repository structure

```text
configs/ten_hour.yaml              # Primary experiment config (375 val cases)
src/analysis/                      # Probing, editing, probe-screen modules
src/training/                      # Inference + cached downstream editing
learn_semantic_directions.py       # Fit Ridge probes and lift edit directions
analyze_representation_editing.py  # Full editing experiment
analyze_edema_probe_screen.py    # decoder1 edema vs random (30 cases)
analyze_volume_probe_screen.py   # decoder2 volume vs random (30 cases)
analyze_layer_interventions.py   # Whole-layer ablation

outputs_10hour/                    # Curated results tracked in git (see below)
├── failure_tables/failure_metrics.csv
├── layer_holdout_recoverability/  # Fold-safe probing R² tables + figures
├── layer_analysis/                # Layer recoverability summary
├── semantic_directions/directions/
├── representation_editing/        # Editing CSVs, figures, report
├── edema_probe_screen/
└── volume_probe_screen/
```

Large artifacts (checkpoints, raw predictions, embeddings, caches) are **not** in git. Train locally or download the checkpoint separately.

---

## Reproducing the analysis

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download [BraTS 2021 training data](https://www.synapse.org/#!Synapse:syn27046444) and set `data.root` in `configs/ten_hour.yaml`.

Train the model (or place a compatible checkpoint at `outputs_10hour/checkpoints/checkpoint_latest.pt`):

```bash
python train.py --config configs/ten_hour.yaml
```

### Probing → editing → probe screens

After training and exporting layer embeddings (`export_layer_embeddings.py`), run:

```bash
bash scripts/run_representation_analysis.sh
```

Or step by step:

```bash
# 1. Learn semantic directions (requires layer embeddings + failure table)
python learn_semantic_directions.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --layer-index outputs_10hour/layer_embeddings/layer_index.csv \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --output-dir outputs_10hour/semantic_directions

# 2. Representation editing (long-running on 375 cases)
python analyze_representation_editing.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --directions-dir outputs_10hour/semantic_directions \
  --output-dir outputs_10hour/representation_editing

# 3. Edema probe screen (30 stratified cases)
python analyze_edema_probe_screen.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv

# 4. Volume probe screen (30 stratified cases, 5 random directions)
python analyze_volume_probe_screen.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --n-random 5
```

Precomputed result tables and figures for the primary 10-hour run are included under `outputs_10hour/` so findings can be inspected without re-running inference.

---

## Citation

If you use this code in academic work, please cite the repository and acknowledge the BraTS challenge data.

BraTS data: Menze et al., CVPR 2015; Bakas et al., 2017, 2018, 2021.

---

## License

Code is released for research purposes. BraTS data use is subject to the [Synapse terms](https://www.synapse.org/#!Synapse:syn27046444).
