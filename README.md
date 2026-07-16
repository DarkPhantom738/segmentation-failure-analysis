# Anatomical Representation–Output Consistency Improves Confidence-Based Failure Triage in 3D Brain Tumor Segmentation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: BraTS](https://img.shields.io/badge/dataset-BraTS%202021-green.svg)](https://www.synapse.org/#!Synapse:syn27046444)

Brain-tumor segmentation models can look great on average and still mess up individual cases. You often will not know which predictions need a second look until someone checks them against a reference mask — and at inference time you do not have that mask.

This repo asks two connected questions about a 3D U-Net on BraTS 2021:

1. **What do the hidden layers actually know?** Can we read anatomy out of them, perturb them, or steer the mask with probe directions?
2. **If steering fails, can the layers still help with quality control?** Specifically: when the network’s internal picture of the tumor disagrees with what the predicted mask says, does that flag bad segmentations better than confidence alone?

**Short answer:** layers carry a lot of useful information, but they are much better for **flagging risky cases** than for **reliably editing** the segmentation. The practical win is failure triage: rank the 375 validation cases for human review using confidence plus representation–output consistency.

Exploratory probing, editing, and repair code lives under [`extra/`](extra/README.md). You do not need it to reproduce the main triage result.

---

## The two threads (how to read this repo)

| | RQ1 — inside the network | RQ2 — failure triage (main result) |
|---|---|---|
| **Question** | Recoverable vs controllable? | Does consistency beat confidence for review ranking? |
| **What we do** | Probes, spatial ablation, probe-aligned edits | Compare anatomy implied by hidden states vs anatomy measured from the **predicted** mask |
| **Headline** | Decodable ≠ controllable | Confidence + consistency improves detection of bad overall segmentations |
| **Where in repo** | [`extra/`](extra/README.md), `src/analysis/*` | `scripts/run_*`, `results/paper/` |

---

## Main result (RQ2)

**375** held-out BraTS cases · nested CV (5 outer / 4 inner folds) · **5000** paired bootstrap replicates · primary target = lowest **20%** by mean foreground Dice.

**Detecting the lowest-quality 20% of segmentations:**

| Method | AUPRC | AUROC | Capture @ 20% review | Brier |
|---|---:|---:|---:|---:|
| Confidence only (baseline) | 0.805 | 0.931 | 71.4% | 0.102 |
| Confidence + morphology | 0.851 | 0.954 | 75.3% | 0.077 |
| Confidence + pooled representations | 0.854 | 0.948 | 74.0% | 0.081 |
| **Confidence + representation–output consistency** | **0.895** | **0.960** | **79.2%** | **0.064** |

**Proposed vs confidence only** (paired bootstrap):

| Metric | Δ | 95% CI | P(proposed better) |
|---|---:|---|---:|
| AUPRC | **+0.090** | **[0.031, 0.152]** | 99.8% |
| Capture @ 20% review | **+0.086** | **[0.013, 0.165]** | 97.7% |
| Mean Dice @ 80% coverage | +0.004 | [0.000, 0.009] | 98.2% |
| AURC (lower is better) | −0.001 | [−0.008, +0.006] | 44.5% |

**Secondary endpoint — mean foreground Dice &lt; 0.70:** AUPRC **0.770** (confidence) → **0.887** (confidence + consistency); ΔAUPRC **+0.115** [0.056, 0.185]. The combined model wins **5/5** outer folds on both primary and secondary endpoints.

These numbers are about **research QC** (which cases to review), not clinical deployment or automatic repair.

Committed tables: [`results/paper/triage_20260712/`](results/paper/triage_20260712/), [`results/paper/method_validation/`](results/paper/method_validation/).

---

## RQ1 in plain language

We wanted to know whether “the network encodes edema fraction / tumor size / Dice” means “you can push the network along that axis and get a cleaner mask.” Mostly, it does not.

### What showed up

| Question | Answer | Example numbers |
|---|---|---|
| Can we read anatomy from hidden layers? | **Often yes** | Whole-tumor volume R² ≈ 0.82; edema fraction ≈ 0.60; enhancing fraction ≈ 0.65; Dice ≈ 0.57 (best layer varies by target) |
| Does wiping out spatial structure hurt? | **Yes, especially late decoder** | Mean replacement at `decoder1` nearly kills the mask; at the bottleneck it barely moves Dice |
| Can probe directions steer the output? | **Weak and inconsistent** | Edema edits beat random directions a bit; highly recoverable tumor-volume direction did **not** reliably move the final mask |

### The story in one contrast

- **Encouraging:** editing `decoder1` along the edema-fraction probe nudged edema fraction more than matched random directions (mean |Δ| 0.021 vs 0.009; ratio ~2.4×).
- **Sobering:** tumor volume was easy to decode (R² 0.82) but hard to control — probe readouts flipped sign as expected far more often than the **output mask** did.

So: **probes are good microscopes, bad steering wheels.** That is why we pivot to consistency for triage instead of representation editing for repair.

**Code paths:** [`extra/README.md`](extra/README.md) · [`extra/scripts/README.md`](extra/scripts/README.md) · probing/editing modules in [`src/analysis/`](src/analysis/README.md).

---

## What “representation–output consistency” means here

For something like enhancing-tumor fraction:

1. Inside each training fold, fit a Ridge probe on a hidden layer (ground truth used only here).
2. Measure the same property from the **predicted** mask (no GT at inference).
3. Take the gap between those two views and feed it, with TTA confidence features, into a regularized logistic model (nested CV).

At review time the detector never sees reference segmentations. GT is only for probe fitting inside folds, defining failure labels, and scoring held-out performance.

Most of the consistency gain comes from **enhancing-fraction** gaps — see `results/paper/method_validation/feature_permutation_importance.csv`. Consistency helps **overall** bad segmentations more than **edema-only** failures (edema endpoint: only ~+0.006 ΔAUPRC vs confidence).

---

## Setup (one glance)

| | |
|---|---|
| Data | BraTS 2021 — 1,251 cases → **876 train / 375 val** (`seed=42`, 30% val) |
| Inputs | T1, T1ce, T2, FLAIR |
| Model | 4-class 3D U-Net; paper checkpoint = **epoch 5** (`configs/ten_hour.yaml`) |
| Layers | Nine stages: `encoder1`–`encoder4`, `bottleneck`, `decoder4`–`decoder1` |
| Robustness track | Converged multi-seed training: `train_converged.py` + `configs/converged_unet.yaml` |

Split verification: [`docs/manuscript/environment_and_split.md`](docs/manuscript/environment_and_split.md).

---

## Figures

**Precision–recall (failure detection):**

![Precision–recall curves](results/paper/triage_20260712/figures/precision_recall_curves.png)

**How many failures you catch if you only review the riskiest 10–30% of cases:**

![Failure capture curves](results/paper/method_validation/figures/failure_capture_curves.png)

**Bootstrap: is the AUPRC gain real?**

![Bootstrap delta AUPRC](results/paper/method_validation/figures/bootstrap_delta_auprc.png)

**What actually drives the detector (enhancing-fraction gaps dominate):**

![Feature permutation importance](results/paper/method_validation/figures/feature_permutation_importance.png)

| If you want… | Open… |
|---|---|
| Full triage numbers | `results/paper/triage_20260712/aggregate_metrics.csv` |
| Bootstrap / ablations write-up | `results/paper/method_validation/validation_summary.md` |
| Per-case features | `results/paper/confidence_features.csv`, `results/paper/consistency/case_level_features.csv` |

---

## Navigate the repo

Every major folder has a README with file roles and pipeline order — you should not have to grep blindly.

| Start here | Why |
|---|---|
| [`docs/paper_pipeline.md`](docs/paper_pipeline.md) | Stage-by-stage: train → TTA → embeddings → triage |
| [`configs/README.md`](configs/README.md) | Which YAML drives which step |
| [`scripts/README.md`](scripts/README.md) | Runnable entry points |
| [`src/analysis/README.md`](src/analysis/README.md) | Where the triage math lives |
| [`results/paper/README.md`](results/paper/README.md) | Committed snapshot layout |

**Quick tour:** this README → `docs/paper_pipeline.md` → `results/paper/` → `scripts/` + `configs/` → `src/` only if you are implementing.

```text
train.py / train_converged.py     # train + export
analyze_failures.py               # case-level Dice / failure table
export_layer_embeddings.py        # layer activations for probes
scripts/run_paper_pipeline.sh     # full paper regen (heavy)
results/paper/                    # numbers + figures in git
extra/                            # RQ1 only
```

Large runs (`outputs_*`, checkpoints, caches) stay **local and gitignored**.

---

## Reproduction

**Environment**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**No GPU?** Browse `results/paper/triage_20260712/` and `results/paper/method_validation/validation_summary.md`.

**Full regen (needs BraTS + local checkpoint):**

```bash
bash scripts/run_paper_pipeline.sh
# or SKIP_TTA=1 if TTA artifacts already exist
```

Step-by-step commands: [`scripts/README.md`](scripts/README.md).

**Multi-seed converged models:**

```bash
bash scripts/run_converged_seeds.sh
bash scripts/run_seed_downstream.sh 123   # after TTA for that seed
```

---

## Honest limitations

- One public dataset, one U-Net family, retrospective labels — not external validation or a clinical workflow study.
- Consistency is a **complement** to confidence, not a replacement; alone it underperforms confidence.
- Biggest signal is enhancing-tumor composition mismatch; edema-specific failures see little gain.
- Risk–coverage AURC did not improve in a statistically supported way.
- Representation editing and repair experiments did not yield a usable auto-fix.

---

## Data governance

BraTS 2021 is subject to [Synapse terms](https://www.synapse.org/#!Synapse:syn27046444). Committed CSVs have no patient IDs. References: Menze et al., CVPR 2015; Bakas et al., 2017–2021.

No `LICENSE` file yet.
