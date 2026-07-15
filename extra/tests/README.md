# Extra tests (`extra/tests/`)

Optional pytest coverage for exploratory repair code. Not part of the minimal paper-reproduction test set under [`../../tests/`](../../tests/README.md).

## Files

| File | Focus |
|---|---|
| `test_spatial_edema_repair.py` | Unit-level checks around `src.models.spatial_edema_repair` (directions, masks, small forward contracts). |

## How to run

```bash
pytest extra/tests -q
```

## Roadmap

Add tests when repair APIs stabilize; keep them isolated so CI can remain light for the main triage package.
