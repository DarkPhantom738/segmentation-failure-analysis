"""I/O helpers for saving validation artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_dir(path: Path | str) -> Path:
    """Create directory if it does not exist and return the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_array(array: np.ndarray, path: Path | str) -> str:
    """Save a NumPy array to disk and return the path as a string."""
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, array)
    return str(path)
