# Source package (`src/`)

Installable library (`pip install -e .`) that implements models, data, training, and analysis.
**Prefer starting from `scripts/`** for runnable stages; use this map when you need to know *which module owns which behavior*.

---

## Package layout

| Subpackage | README | Responsibility |
|---|---|---|
| [`data/`](data/README.md) | BraTS discovery, preprocessing, dataloaders, converged shared splits |
| [`models/`](models/README.md) | 3D U-Net (+ optional exploratory repair modules) |
| [`training/`](training/README.md) | Train loops, sliding-window inference, TTA, metrics, specialty inference for editing/repair |
| [`analysis/`](analysis/README.md) | Confidence, consistency, triage nested CV, method validation, RQ1 helpers |
| [`utils/`](utils/README.md) | Seed setting, small I/O helpers |

`src/__init__.py` marks the package; it does not re-export the public API.

---

## Dependency direction (how to navigate)

```text
scripts / train*.py
    ↓
src.training  +  src.analysis
    ↓              ↓
src.models    src.data
    ↓              ↓
src.utils
```

Analysis modules generally **read artifacts** (CSVs, embeddings, predictions) rather than re-implement U-Net training. Training never depends on triage math.

---

## Two research tracks inside one package

| Track | Question | Code concentration | Required for main README table? |
|---|---|---|---|
| **RQ2 triage** | Does confidence + representation–output consistency beat confidence alone for failure ranking? | `analysis/layer_aware_*`, `representation_output_consistency`, `confidence_consistency_triage`, `method_validation` | **Yes** |
| **RQ1 recoverability / control** | Are anatomical properties decodable? Do edits/repairs control the mask? | `analysis/semantic_directions`, `representation_editing`, `layer_interventions`, `*_probe_screen`, `models/*repair*`, `training/*editing*` / `*repair*` | **No** (CLIs under `extra/`) |

---

## Conventions that appear everywhere

1. **Nine named layers** (`encoder1` … `decoder1`) — defined in `models/unet3d.py` as `LAYER_NAMES`; embedding export and probes use the same names.
2. **BraTS label remap** `{0,1,2,4} → {0,1,2,3}` — in `data/preprocessing.py`; all Dice/metrics assume remapped labels.
3. **Leakage rule for detectors** — GT may fit probes **inside training folds**, define labels, and score held-out metrics; GT-derived features must **not** be detector inputs at inference. Enforced in analysis modules (see their README).
4. **Historical CSV column names** — some columns say `out_gt_*` even when the value is measured from the **predicted** mask; do not rename casually (breaks configs and committed tables).

---

## Roadmap (library)

| Status | Item |
|---|---|
| Frozen | Triage / consistency algorithms used for `results/paper/` |
| Active | Converged multi-seed training code path |
| Optional extras | Repair / editing remain available but are not the product claim |
| Explicit non-goals | Replacing sliding-window/TTA with third-party stacks in a way that changes numerics |

Drill down: open the README inside the subpackage you care about.
