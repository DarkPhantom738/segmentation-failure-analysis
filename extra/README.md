# Extra / exploratory analyses (`extra/`)

RQ1 probing, representation editing, layer interventions, and repair **feasibility** experiments.

**Not required** to reproduce the root README failure-triage table (confidence + representation–output consistency on 375 cases). Core path: root [`README.md`](../README.md) → [`docs/paper_pipeline.md`](../docs/paper_pipeline.md).

Library code stays under `src/` (no algorithm forks here)—these are CLIs, YAML, notes, and optional tests.

---

## Folder map

| Path | README | Contents |
|---|---|---|
| [`scripts/`](scripts/README.md) | CLI wrappers for recoverability, directions, editing, interventions, repair |
| [`configs/`](configs/README.md) | YAML for fast spatial / classifier-inverted repair |
| [`docs/`](docs/README.md) | Author reporting notes (wording + claim hygiene) |
| [`tests/`](tests/README.md) | Optional unit tests for repair modules |

---

## Scientific role (RQ1)

| Question | What this folder runs | Typical conclusion style |
|---|---|---|
| Are anatomical properties recoverable from layers? | Holdout recoverability, semantic directions | Often **yes (decodable)** |
| Does probe-aligned editing control the mask? | Representation editing + matched-random screens | Often **weak / non-specific control** |
| Can a tiny repair head fix edema errors? | Spatial / classifier-inverted repair smokes | Feasibility only—not a deployment claim |

Wording guide: [`docs/reporting_notes.md`](docs/reporting_notes.md).

---

## Prerequisites

```bash
# repo root
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

You typically need:

- BraTS root or local preprocessed cache (`outputs_10hour/cache/`)
- Checkpoint (`outputs_10hour/checkpoints/checkpoint_latest.pt` or equivalent)
- Case table (`results/paper/failure_metrics.csv` and/or local failure tables)
- Layer index/embeddings if running recoverability / directions (`outputs_10hour/layer_embeddings/`)

Outputs write under gitignored `outputs_*`.

---

## Quick start

```bash
bash extra/scripts/run_representation_analysis.sh
```

Or step-through commands listed in [`scripts/README.md`](scripts/README.md).

Repair smoke:

```bash
python extra/scripts/run_fast_spatial_edema_repair.py \
  --config extra/configs/fast_spatial_edema_repair.yaml --stage smoke
```

---

## Roadmap

| Status | Item |
|---|---|
| Available | RQ1 screens + repair feasibility CLIs |
| Separate from product claim | Do not treat repair Dice gains as the paper endpoint |
| Future | Only expand if pursuing a dedicated RQ1/repair paper section |

## Related

- Analysis library: [`src/analysis/README.md`](../src/analysis/README.md)
- Model repair modules: [`src/models/README.md`](../src/models/README.md)
