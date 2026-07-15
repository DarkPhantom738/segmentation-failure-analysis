# Paper pipeline guide

This document is a reviewer-facing map of the code path used for the paper results.
It is descriptive only; it does not introduce a separate pipeline from the scripts in
`scripts/`.

## What is committed

The GitHub tree intentionally commits only compact, reviewer-facing artifacts:

- code under `src/`
- command wrappers under `scripts/`
- YAML configs under `configs/`
- paper tables and figures under `results/paper/`

Large local artifacts are ignored by Git:

- model checkpoints
- preprocessed caches
- layer embedding arrays
- hard prediction arrays
- softmax/probability volumes
- full `outputs_*` run folders

The committed paper snapshot is enough to inspect the reported tables and figures.
Regenerating the full pipeline requires the BraTS 2021 data and local checkpoints.

## Main result path

The canonical result in the root README is the epoch-5/10-hour checkpoint analysis.
The end-to-end orchestration script is:

```bash
bash scripts/run_paper_pipeline.sh
```

That script executes the following stages:

| Stage | Command or module | Purpose | Main local output |
|---|---|---|---|
| 1 | `train.py --export-tta` | Frozen TTA inference from the checkpoint; writes hard predictions and uncertainty metrics | `outputs_10hour/` |
| 2 | `analyze_failures.py` | Computes case-level Dice/failure labels from segmentation outputs | `outputs_10hour/failure_tables/` |
| 3 | `export_layer_embeddings.py` | Extracts globally pooled activations for the nine analyzed network stages | `outputs_10hour/layer_embeddings/` |
| 4 | `scripts/run_layer_aware_latent_risk.py --stage confidence` | Recomputes compact GT-free confidence summaries and checks hard-mask consistency | `outputs_layer_aware_latent_risk/` |
| 5 | `scripts/run_consistency_failure_detection.py` | Feasibility analysis for representation-output gaps and related baselines | `outputs_consistency_failure_detection/` |
| 6 | `scripts/run_confidence_consistency_triage.py` | Main nested-CV confidence plus consistency triage model | `outputs_confidence_consistency_triage*/` |
| 7 | `scripts/run_method_validation.py` | Final validation package: bootstrap, ablations, feature importance, and figures | `outputs_method_validation/` |

The committed snapshot corresponding to these results is:

- `results/paper/triage_20260712/`
- `results/paper/method_validation/`
- `results/paper/confidence_features.csv`
- `results/paper/consistency/case_level_features.csv`
- `results/paper/failure_metrics.csv`
- `results/paper/fold_assignments.csv`

## Converged seed-42 replication

The converged seed-42 model is a robustness/replication analysis, not a replacement
for the canonical epoch-5 table. Its compact committed snapshot is:

```text
results/paper/converged_seed042/
```

The local downstream chain used for converged seeds is:

```bash
bash scripts/run_seed_downstream.sh 42
```

That script reuses the same downstream analysis code but retargets the output paths
to `outputs_converged/seed_042/analysis/`.

## What is optional or exploratory

The following are not required to reproduce the main confidence-plus-consistency
triage result:

- `extra/` scripts and configs
- representation editing screens
- spatial repair / classifier repair experiments
- layer intervention and semantic-direction scripts used for exploratory RQ1 work

They are retained for transparency, but the main reviewer path starts with
`scripts/run_paper_pipeline.sh` and the committed tables in `results/paper/`.

## Leakage rules used by the pipeline

The failure detector must not receive ground-truth-derived features at inference
time. The code enforces and documents these rules in the relevant analysis modules:

- Confidence features are computed from model outputs only.
- Ground truth is used to fit anatomical Ridge probes inside training folds.
- Ground truth is used to define failure labels and evaluate held-out predictions.
- GT-linked uncertainty/error columns from failure tables are blocked as detector features.
- Percentile-based failure thresholds are estimated inside the outer-training fold and then applied to the held-out fold.

These rules are implemented mainly in:

- `src/analysis/layer_aware_latent_risk.py`
- `src/analysis/representation_output_consistency.py`
- `src/analysis/confidence_consistency_triage.py`
- `src/analysis/method_validation.py`
