# Utilities (`src/utils/`)

Tiny shared helpers with no research logic.

## Files

| File | What it does |
|---|---|
| `__init__.py` | Package marker. |
| `seed.py` | `set_seed(...)` — seeds Python `random`, NumPy, and PyTorch (and related determinism knobs used by train entrypoints). Call this at the start of training / export jobs. |
| `io.py` | Filesystem helpers such as `ensure_dir` used when writing analysis / embedding outputs. |

## Roadmap

Keep this package small. Prefer adding domain code under `analysis/`, `training/`, or `data/` rather than growing a catch-all utils namespace.
