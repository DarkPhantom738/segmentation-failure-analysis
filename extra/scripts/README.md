# Extra scripts (`extra/scripts/`)

Command-line entry points for RQ1 and repair experiments. Each file is thin relative to the `src/analysis` / `src/models` / `src/training` libraries it calls.

**Paper triage CLIs are not here** — see [`../../scripts/README.md`](../../scripts/README.md).

---

## Orchestration

| Script | Role |
|---|---|
| `run_representation_analysis.sh` | Runs a default sequence of RQ1 analyses (directions → editing → recoverability → interventions) against local `outputs_10hour/` paths. |

---

## RQ1 analysis CLIs

| Script | Library focus | What you get |
|---|---|---|
| `learn_semantic_directions.py` | `src.analysis.semantic_directions` | Fold-safe probes + activation-space edit directions catalog under an output dir. |
| `analyze_representation_editing.py` | `src.analysis.representation_editing` | Edit sweeps measuring on-target vs collateral mask changes. |
| `analyze_layer_holdout_recoverability.py` | `src.analysis.layer_holdout_recoverability` | Locked holdout recoverability metrics / heatmaps. |
| `analyze_layer_interventions.py` | `src.analysis.layer_interventions` | Spatial mean ablation / dependence screens. |
| `analyze_edema_probe_screen.py` | `src.analysis.edema_probe_screen` | Matched-random screen for edema direction at decoder1. |
| `analyze_volume_probe_screen.py` | `src.analysis.volume_probe_screen` | Matched-random screen for tumor-volume direction at decoder2. |

Example (from repo root):

```bash
python extra/scripts/learn_semantic_directions.py \
  --config configs/ten_hour.yaml \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt \
  --layer-index outputs_10hour/layer_embeddings/layer_embedding_index.csv \
  --failure-table results/paper/failure_metrics.csv \
  --output-dir outputs_10hour/semantic_directions
```

---

## Repair feasibility CLIs

| Script | Config | Role |
|---|---|---|
| `run_fast_spatial_edema_repair.py` | `extra/configs/fast_spatial_edema_repair.yaml` | Train/eval spatially gated edema repair (`--stage smoke` for a short path). |
| `run_fast_classifier_inverted_repair.py` | `extra/configs/fast_classifier_inverted_repair.yaml` | Classifier-inverted transition-gate repair stages. |
| `run_decoder1_repair_diagnostics.py` | Reuses a repair yaml | Diagnostics around decoder1 caches / repair behavior. |

These write under local `outputs_fast_*` / related gitignored dirs.

---

## Roadmap

| Status | Item |
|---|---|
| Stable enough for transparency | Scripts matching analyses discussed in RQ1 narrative |
| Not productized | No guarantee of bit-identical repair tables across machines without locked seeds + cache |
| Keep separate | Do not fold repair stages into `scripts/run_paper_pipeline.sh` |

## Related

- Parent: [`../README.md`](../README.md)
- Reporting language: [`../docs/reporting_notes.md`](../docs/reporting_notes.md)
