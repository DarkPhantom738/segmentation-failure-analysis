# Configuration files (`configs/`)

YAML knobs for training and the paper analysis chain. Each config is consumed by a thin CLI in `scripts/` or a root entrypoint (`train.py`, `train_converged.py`, `export_layer_embeddings.py`).

**Important:** Changing paths retargets I/O. Changing nested-CV folds, feature lists, seeds, or scoring definitions can change science results. For replication of published tables, keep the canonical configs as committed unless you intentionally start a new experiment.

---

## How to choose a config

```text
Want the numbers in the root README table?
  → epoch-5 / "ten hour" path:
      ten_hour.yaml
      layer_aware_latent_risk.yaml
      consistency_failure_detection.yaml
      confidence_consistency_triage.yaml
      method_validation.yaml
  Orchestrator: scripts/run_paper_pipeline.sh

Want multi-seed converged U-Net robustness?
  → converged_unet.yaml + train_converged.py
  → then seed templates (converged_seed042_*.yaml) via
      scripts/run_seed_downstream.sh <seed>
```

Full stage narrative: [`docs/paper_pipeline.md`](../docs/paper_pipeline.md).

---

## File-by-file reference

### Canonical epoch-5 (paper main result)

| File | Entry point | What it controls | Typical outputs |
|---|---|---|---|
| `ten_hour.yaml` | `train.py`, `export_layer_embeddings.py` | Data root, modalities, spacing, patch size, U-Net channels, optimizer, 876/375 split seeds, checkpoint/export paths for the original short training regime (paper checkpoint = epoch 5). | Local `outputs_10hour/` (gitignored) |
| `layer_aware_latent_risk.yaml` | `scripts/run_layer_aware_latent_risk.py` | Paths to predictions / failure table / embeddings; stages (`artifact`, `confidence`, …); confidence feature definitions; leakage guards. Paper snapshot inputs often redirect to `results/paper/`. | Local `outputs_layer_aware_latent_risk/`; committed `results/paper/confidence_features.csv` |
| `consistency_failure_detection.yaml` | `scripts/run_consistency_failure_detection.py` | Consistency feasibility: which layers/properties form gaps, nested-CV settings, baselines vs morphology/pooled reps. | Local consistency dirs; committed `results/paper/consistency/` |
| `confidence_consistency_triage.yaml` | `scripts/run_confidence_consistency_triage.py` | Main triage evaluation: endpoints (lowest 20%, Dice&lt;0.70), nested CV (outer/inner), logistic grids, bootstrap count, feature bundles (conf / morph / pooled / consistency / combined). | Local triage run dirs; committed `results/paper/triage_20260712/` |
| `method_validation.yaml` | `scripts/run_method_validation.py` | Validation package on frozen triage scores: bootstrap comparisons, ablations, permutation importance, risk–coverage, capture@budget, figures. | Local validation dirs; committed `results/paper/method_validation/` |

### Converged multi-seed replication

| File | Entry point | What it controls |
|---|---|---|
| `converged_unet.yaml` | `train_converged.py` | Shared converged-training protocol: same patient splits across seeds, development early-stop metric, LR schedule, output under `outputs_converged/seed_*/`. Does **not** write into `outputs_10hour/`. |
| `converged_seed042_export.yaml` | `train.py --export-tta` | TTA export paths for the converged seed-42 checkpoint (reuses the same TTA code path as the paper export). |
| `converged_seed042_layer_aware.yaml` | `run_layer_aware_latent_risk.py` | Seed-42 confidence stage paths (templates). |
| `converged_seed042_consistency.yaml` | `run_consistency_failure_detection.py` | Seed-42 consistency stage paths. |
| `converged_seed042_triage.yaml` | `run_confidence_consistency_triage.py` | Seed-42 triage paths/endpoints. |
| `converged_seed042_method_validation.yaml` | `run_method_validation.py` | Seed-42 validation package paths. |

`scripts/run_seed_downstream.sh <seed>` copies these 042 templates, rewrites output directories to `outputs_converged/seed_XXX/analysis/`, and runs the chain. **Algorithms stay the same; only paths change.**

---

## Path conventions

| Path style | Meaning |
|---|---|
| `results/paper/...` | Committed snapshot (safe to open in a clone with no GPU). |
| `outputs_*` | Local regeneration (gitignored). New runs should **write here**, not overwrite `results/paper/` casually. |
| `outputs_10hour/` | Historical paper training + TTA + embeddings (local). |
| `outputs_converged/` | Multi-seed converged models (local). |

---

## Roadmap

| Status | Item |
|---|---|
| Frozen | Canonical yaml used for `results/paper/triage_20260712` and `method_validation` |
| In progress (research) | Remaining converged seeds (123, 2026, …) via seed downstream script |
| Optional | New experiment configs → add new files; do not silently edit canonical ones |
| Avoid | Pointing paper configs at temporary debug dirs without documenting it |

## Related docs

- [`scripts/README.md`](../scripts/README.md)
- [`results/paper/README.md`](../results/paper/README.md)
- [`docs/manuscript/environment_and_split.md`](../docs/manuscript/environment_and_split.md)
