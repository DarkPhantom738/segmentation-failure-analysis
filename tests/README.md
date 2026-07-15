# Tests (`tests/`)

Pytest suite for **behavioral guards** around the paper analysis and converged-training helpers. These tests do not replace a full GPU BraTS run; they catch regressions in fold logic, leakage rules, metric wiring, and small unit contracts.

## How to run

```bash
# from repo root, with editable install
pip install -e ".[dev]"
pytest
# or a single file:
pytest tests/test_confidence_consistency_triage.py -q
```

Configured via `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["."]`).

---

## Files

| File | What it protects |
|---|---|
| `test_confidence_consistency_triage.py` | Helpers and contracts in the main triage module (metrics, impute/scale/tune paths, leakage-sensitive pieces that can be unit-tested without full nested CV on 375 cases). |
| `test_consistency_failure_detection.py` | Representation–output consistency helpers: gap construction, GT-leak column discipline, probe / feature utilities used by the feasibility stage. |
| `test_layer_aware_latent_risk.py` | Confidence-feature / artifact-check utilities in the layer-aware module (GT-free feature expectations). |
| `test_converged_training.py` | Shared-split helpers, seed directory naming, and converged-config loading / protocol pieces used by `train_converged.py`. |

There is no large end-to-end “reproduce the paper AUPRC on CI” test; that requires local BraTS + checkpoints + hours of compute. Committed CSVs under `results/paper/` are the reference artifacts for those numbers.

Optional repair coverage lives under [`extra/tests/`](../extra/tests/README.md).

---

## Roadmap

| Status | Item |
|---|---|
| Present | Unit/integration guards for core analysis + converged split protocol |
| Good additions | Golden-file smoke tests that load tiny synthetic arrays through gap-feature helpers |
| Avoid | Tests that silently rewrite algorithms to make them “easier to mock” |

## Related

- Package under test: [`src/analysis/README.md`](../src/analysis/README.md), [`src/data/README.md`](../src/data/README.md)
- Orchestration: [`scripts/README.md`](../scripts/README.md)
