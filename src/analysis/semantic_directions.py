"""
Learn semantic probe directions and lift them into activation tensor space.

Mathematical basis (adjoint of global average pooling)
------------------------------------------------------

Probes are trained on pooled readouts ``v = GAP(A)`` (or ``e = W @ GAP(A)`` at
bottleneck). A linear probe with fold-safe standardization predicts property
``y`` from scaled embedding ``x_s = (x - mu) / sigma``:

    y_hat = beta^T x_s + b

The learned coefficient is first converted from standardized probe space back
to the unscaled pooled-readout coordinates. The **minimum L2-norm**
perturbation ``Delta`` to the activation map ``A`` satisfying a desired pooled
shift ``GAP(A + Delta) - GAP(A) = delta x`` is spatially constant per channel:

    Delta_{c,d,h,w} = delta x_c

This is the minimum-norm lift associated with global average pooling and
matches activation-steering / representation-engineering practice: channel
directions broadcast across space induce controlled shifts in linearly decoded
pooled features.

Bottleneck maps are edited before ``embedding_head``. With ``e = W @ v`` and
probe on ``e``, the minimum-norm lift uses the Moore-Penrose pseudoinverse:

    delta v = W^+ delta e,   Delta_{c,d,h,w} = (delta v)_c

Directions are unit-normalized in activation channel space. Therefore edit
strength ``alpha`` is an activation-space dose along the semantic direction,
not one literal standardized-probe unit.

Encoder editing semantics
-------------------------
Probes on encoder layers use ``GAP(encoder_block_output)``, which is the same tensor
passed as a skip connection. Encoder edits therefore target the **skip path** at
decode fusion, not activations propagated into deeper encoder blocks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.baselines import choose_cv_folds
from src.analysis.layer_analysis import (
    ANATOMY_TARGET_SPECS,
    build_anatomy_table,
    load_layer_index,
    load_layer_matrix,
)
from src.models.unet3d import LAYER_NAMES, UNet3D, build_model
from src.utils.io import ensure_dir

# Properties requested for causal editing (probe target key -> eval metric key).
EDIT_PROPERTY_SPECS: dict[str, str] = {
    "gt_wt_voxels": "tumor_volume",
    "gt_edema_frac": "edema_fraction",
    "gt_enhancing_frac": "enhancing_fraction",
    "gt_necrosis_frac": "necrosis_fraction",
    "gt_compactness": "boundary_complexity",
    "dice": "dice",
    "boundary_error_fraction": "boundary_error_fraction",
}

# Also probe centroid for evaluation / optional directions.
AUXILIARY_PROBE_TARGETS: dict[str, str] = {
    "centroid_z": "centroid_z",
}

DEFAULT_ALPHAS: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


@dataclass
class SemanticDirection:
    """A unit semantic direction in activation tensor space."""

    layer: str
    property_key: str
    property_label: str
    eval_metric: str
    oof_r2: float
    oof_mae: float
    n_cases: int
    cv_folds: int
    probe_coef_scaled: np.ndarray
    probe_intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    gap_delta_unscaled: np.ndarray
    activation_direction: np.ndarray
    readout_type: str
    embedding_dim: int
    activation_channels: int

    def save(self, path: Path) -> None:
        path = Path(path)
        meta = asdict(self)
        for key in list(meta.keys()):
            if isinstance(meta[key], np.ndarray):
                meta[key] = None
        np.savez(
            path,
            probe_coef_scaled=self.probe_coef_scaled.astype(np.float32),
            probe_intercept=np.float32(self.probe_intercept),
            scaler_mean=self.scaler_mean.astype(np.float32),
            scaler_scale=self.scaler_scale.astype(np.float32),
            gap_delta_unscaled=self.gap_delta_unscaled.astype(np.float32),
            activation_direction=self.activation_direction.astype(np.float32),
        )
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, npz_path: Path) -> SemanticDirection:
        npz_path = Path(npz_path)
        meta = json.loads(npz_path.with_suffix(".json").read_text())
        arrays = np.load(npz_path)
        array_keys = {
            "probe_coef_scaled",
            "probe_intercept",
            "scaler_mean",
            "scaler_scale",
            "gap_delta_unscaled",
            "activation_direction",
        }
        for key in array_keys:
            meta.pop(key, None)
        return cls(
            **meta,
            probe_coef_scaled=arrays["probe_coef_scaled"],
            probe_intercept=float(arrays["probe_intercept"]) if "probe_intercept" in arrays else 0.0,
            scaler_mean=arrays["scaler_mean"],
            scaler_scale=arrays["scaler_scale"],
            gap_delta_unscaled=arrays["gap_delta_unscaled"],
            activation_direction=arrays["activation_direction"],
        )


def _unit_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def lift_gap_adjoint(
    scaled_probe_coef: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    """
  Lift a scaled-space probe coefficient to a GAP readout perturbation.

  ``delta v_unscaled = sigma * beta_hat`` where ``beta_hat`` is unit norm in
  scaled probe input space.
  """
    beta_hat = _unit_normalize(scaled_probe_coef)
    return scaler_scale * beta_hat


def lift_bottleneck_adjoint(
    scaled_probe_coef: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    linear_weight: np.ndarray,
) -> np.ndarray:
    """
    Lift bottleneck embedding probe direction to bottleneck-map channel space.

    ``delta e = sigma * beta_hat`` in embedding space, then
    ``delta v = W^+ delta e`` (minimum L2 norm solution).
    """
    beta_hat = _unit_normalize(scaled_probe_coef)
    delta_e = scaler_scale * beta_hat
    w = np.asarray(linear_weight, dtype=np.float64)
    # W shape (embedding_dim, channels); pseudoinverse for underdetermined lift.
    gram = w @ w.T
    delta_v = w.T @ np.linalg.solve(gram + 1e-6 * np.eye(gram.shape[0]), delta_e)
    return delta_v


def activation_direction_from_gap_delta(gap_delta: np.ndarray) -> np.ndarray:
    """Unit-normalize the channel direction used for spatially constant edits."""
    return _unit_normalize(np.asarray(gap_delta, dtype=np.float64))


def _oof_ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int = 42,
) -> tuple[float, float, np.ndarray, float, np.ndarray, np.ndarray]:
    """Fold-safe Ridge; return OOF R², mean fold coefficients, and intercept."""
    valid = np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(y) < max(cv_folds, 5):
        raise ValueError(f"Insufficient samples: {len(y)}")

    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    preds = np.zeros(len(y), dtype=np.float64)
    coefs: list[np.ndarray] = []
    intercepts: list[float] = []
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []

    for train_idx, test_idx in splitter.split(x):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-2, 3, 20))),
            ]
        )
        pipe.fit(x[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(x[test_idx])
        scaler: StandardScaler = pipe.named_steps["scaler"]
        model: RidgeCV = pipe.named_steps["model"]
        coefs.append(model.coef_.astype(np.float64))
        intercepts.append(float(model.intercept_))
        means.append(scaler.mean_.astype(np.float64))
        scales.append(scaler.scale_.astype(np.float64))

    oof_r2 = float(r2_score(y, preds))
    oof_mae = float(np.mean(np.abs(y - preds)))
    mean_coef = np.mean(np.stack(coefs, axis=0), axis=0)
    mean_intercept = float(np.mean(intercepts))
    mean_scaler_mean = np.mean(np.stack(means, axis=0), axis=0)
    mean_scaler_scale = np.mean(np.stack(scales, axis=0), axis=0)
    return oof_r2, oof_mae, mean_coef, mean_intercept, mean_scaler_mean, mean_scaler_scale


def _get_bottleneck_linear_weight(model: UNet3D) -> np.ndarray:
    """Return W from embedding_head Linear (out_features, in_features)."""
    for module in model.embedding_head:
        if isinstance(module, nn.Linear):
            return module.weight.detach().cpu().numpy()
    raise ValueError("Could not find Linear layer in embedding_head")


def learn_semantic_directions(
    layer_index_path: Path,
    failure_table_path: Path,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    config: dict | None = None,
    min_r2: float = 0.15,
    property_keys: tuple[str, ...] | None = None,
    include_auxiliary: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit fold-safe probes and save unit activation-space directions.

    Only (layer, property) pairs with OOF R² >= min_r2 are exported.
    """
    output_dir = ensure_dir(output_dir)
    directions_dir = ensure_dir(output_dir / "directions")

    index_df = load_layer_index(layer_index_path)
    failure_df = pd.read_csv(failure_table_path)
    anatomy_df = build_anatomy_table(index_df, failure_df)

    target_specs: dict[str, str] = dict(EDIT_PROPERTY_SPECS)
    if include_auxiliary:
        target_specs.update(AUXILIARY_PROBE_TARGETS)
    if property_keys is not None:
        target_specs = {k: target_specs[k] for k in property_keys if k in target_specs}

    bottleneck_weight: np.ndarray | None = None
    if checkpoint_path is not None and config is not None:
        model = build_model(config)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        bottleneck_weight = _get_bottleneck_linear_weight(model)

    cv_folds = max(choose_cv_folds(np.ones(len(index_df), dtype=int)), 2)
    rows: list[dict[str, Any]] = []

    for layer_name in LAYER_NAMES:
        embeddings = load_layer_matrix(index_df, layer_name)
        n_channels = embeddings.shape[1]

        for prop_key, eval_metric in target_specs.items():
            label = ANATOMY_TARGET_SPECS.get(prop_key, prop_key)
            y = anatomy_df[prop_key].values.astype(np.float64)
            try:
                oof_r2, oof_mae, coef, intercept, s_mean, s_scale = _oof_ridge_fit(
                    embeddings, y, cv_folds=cv_folds, random_state=random_state
                )
            except ValueError:
                continue

            if oof_r2 < min_r2:
                continue

            if layer_name == "bottleneck":
                if bottleneck_weight is None:
                    raise ValueError(
                        "checkpoint_path and config required for bottleneck direction lift"
                    )
                gap_delta = lift_bottleneck_adjoint(coef, s_mean, s_scale, bottleneck_weight)
                readout_type = "linear_gap"
            else:
                gap_delta = lift_gap_adjoint(coef, s_mean, s_scale)
                readout_type = "gap"

            act_dir = activation_direction_from_gap_delta(gap_delta)
            direction = SemanticDirection(
                layer=layer_name,
                property_key=prop_key,
                property_label=label,
                eval_metric=eval_metric,
                oof_r2=oof_r2,
                oof_mae=oof_mae,
                n_cases=int(len(y)),
                cv_folds=cv_folds,
                probe_coef_scaled=coef,
                probe_intercept=intercept,
                scaler_mean=s_mean,
                scaler_scale=s_scale,
                gap_delta_unscaled=gap_delta,
                activation_direction=act_dir,
                readout_type=readout_type,
                embedding_dim=int(embeddings.shape[1]),
                activation_channels=int(
                    bottleneck_weight.shape[1] if layer_name == "bottleneck" else n_channels
                ),
            )
            fname = f"{layer_name}__{prop_key}.npz"
            direction.save(directions_dir / fname)
            rows.append(
                {
                    "layer": layer_name,
                    "property_key": prop_key,
                    "property_label": label,
                    "eval_metric": eval_metric,
                    "oof_r2": oof_r2,
                    "oof_mae": oof_mae,
                    "n_cases": len(y),
                    "cv_folds": cv_folds,
                    "readout_type": readout_type,
                    "activation_channels": direction.activation_channels,
                    "direction_path": str(directions_dir / fname),
                }
            )

    catalog = pd.DataFrame(rows).sort_values(["property_key", "oof_r2"], ascending=[True, False])
    catalog.to_csv(output_dir / "semantic_directions_catalog.csv", index=False)
    return catalog


def load_direction_catalog(output_dir: Path) -> list[SemanticDirection]:
    """Load all saved semantic directions."""
    catalog_path = Path(output_dir) / "semantic_directions_catalog.csv"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing {catalog_path}")
    catalog = pd.read_csv(catalog_path)
    directions: list[SemanticDirection] = []
    for path in catalog["direction_path"]:
        directions.append(SemanticDirection.load(Path(path)))
    return directions
