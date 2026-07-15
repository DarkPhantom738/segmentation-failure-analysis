# Scripts (`scripts/`)

Thin command-line wrappers and shell orchestrators. **Business logic lives in `src/`**; these files exist so a reader can see *pipeline order* without opening thousand-line modules.

Also see root entrypoints (not in this folder): `train.py`, `train_converged.py`, `analyze_failures.py`, `export_layer_embeddings.py`.

---

## Mental model

```text
Orchestration (shell)
  run_paper_pipeline.sh          → full epoch-5 regeneration
  run_converged_seeds.sh         → train many model seeds
  run_seed_downstream.sh         → analysis chain for one seed
  run_seed042_downstream_after_export.sh → historical seed-42 helper

Python stage CLIs (one job each)
  run_layer_aware_latent_risk.py
  run_consistency_failure_detection.py
  run_confidence_consistency_triage.py
  run_method_validation.py
```

Stage narrative with I/O tables: [`docs/paper_pipeline.md`](../docs/paper_pipeline.md).

---

## File-by-file: paper (epoch-5) path

| Script | Implements / calls | When to run | Notes |
|---|---|---|---|
| `run_paper_pipeline.sh` | Ordered bash: TTA export → failure table → embeddings → confidence → consistency → triage → method validation | Prefer this for full regen | Heavy (GPU/CPU + disk). `SKIP_TTA=1` skips export if local TTA artifacts already exist. |
| `run_layer_aware_latent_risk.py` | `src.analysis.layer_aware_latent_risk` | Stage 4 (confidence features / artifact checks) | Supports `--stage` (e.g. `confidence`). Config: `configs/layer_aware_latent_risk.yaml`. |
| `run_consistency_failure_detection.py` | `src.analysis.representation_output_consistency` | Stage 5 (gap-feature feasibility) | Config: `configs/consistency_failure_detection.yaml`. |
| `run_confidence_consistency_triage.py` | `src.analysis.confidence_consistency_triage` | Stage 6 (**main nested-CV triage**) | Config: `configs/confidence_consistency_triage.yaml`. Produces the metrics that land in `results/paper/triage_20260712/`. |
| `run_method_validation.py` | `src.analysis.method_validation` | Stage 7 (bootstrap, ablations, figures) | Config: `configs/method_validation.yaml`. Snapshot: `results/paper/method_validation/`. |

### Typical stage-only invocation

```bash
python scripts/run_layer_aware_latent_risk.py \
  --config configs/layer_aware_latent_risk.yaml --stage confidence

python scripts/run_consistency_failure_detection.py \
  --config configs/consistency_failure_detection.yaml

python scripts/run_confidence_consistency_triage.py \
  --config configs/confidence_consistency_triage.yaml

python scripts/run_method_validation.py \
  --config configs/method_validation.yaml
```

---

## File-by-file: converged multi-seed path

| Script | Role | Detail |
|---|---|---|
| `run_converged_seeds.sh` | Train seeds listed for `train_converged.py` | Skips a seed if `convergence_summary.json` already exists under that seed’s output dir. |
| `run_seed_downstream.sh` | Full downstream after TTA for **one** seed | Retargets `converged_seed042_*.yaml` paths into `outputs_converged/seed_XXX/analysis/`; runs confidence → consistency → triage → method validation. **Does not change algorithms.** |
| `run_seed042_downstream_after_export.sh` | Convenience wrapper for seed 42 | Kept for reproducibility of the completed local seed-42 chain. |

```bash
bash scripts/run_converged_seeds.sh
python train_converged.py --config configs/converged_unet.yaml --seed 123
# after TTA export for that seed:
bash scripts/run_seed_downstream.sh 123
```

---

## Root entrypoints (siblings of this folder)

| File | Role in the roadmap |
|---|---|
| `../train.py` | Train original U-Net **or** `--export-only` / `--export-tta` from a checkpoint (sliding-window + 8-fold flip TTA). |
| `../train_converged.py` | Converged training with shared splits + development early stopping. |
| `../analyze_failures.py` | Build `failure_metrics.csv` from TTA export metrics (`src.analysis.analyze`). |
| `../export_layer_embeddings.py` | Globally pooled activations for the nine U-Net stages (`src.analysis.layer_export`). |

Exploratory RQ1 / repair CLIs: [`extra/scripts/README.md`](../extra/scripts/README.md).

---

## Roadmap

| Status | Item |
|---|---|
| Stable | Paper orchestrator + four Python stage CLIs |
| Active | Multi-seed converged training + per-seed downstream |
| Later | Multi-seed meta-summary script (aggregating `outputs_converged/seed_*/analysis/`) |
| Out of scope here | Editing `src/analysis` algorithms “for convenience” |

## Related docs

- [`configs/README.md`](../configs/README.md)
- [`src/analysis/README.md`](../src/analysis/README.md)
- [`results/paper/README.md`](../results/paper/README.md)
