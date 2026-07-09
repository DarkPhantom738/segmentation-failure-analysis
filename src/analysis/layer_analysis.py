"""Multi-layer anatomical recoverability and quality-estimation analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.baselines import (
    _impute_nan,
    choose_cv_folds,
    compute_deployable_uncertainty_features,
    create_bad_case_label,
)
from src.models.unet3d import LAYER_NAMES
from src.training.metrics import whole_tumor_mask
from src.utils.io import ensure_dir

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
}

LAYER_ORDER = list(LAYER_NAMES)

ANATOMY_TARGET_SPECS: dict[str, str] = {
    "centroid_x": "Centroid X",
    "centroid_y": "Centroid Y",
    "centroid_z": "Centroid Z",
    "gt_wt_voxels": "Whole-tumor volume",
    "log_wt_volume": "log(WT volume)",
    "gt_edema_voxels": "Edema volume",
    "gt_enhancing_voxels": "Enhancing volume",
    "gt_necrosis_voxels": "Necrosis volume",
    "gt_edema_frac": "Edema fraction",
    "gt_enhancing_frac": "Enhancing fraction",
    "gt_necrosis_frac": "Necrosis fraction",
    "gt_compactness": "Boundary complexity",
    "boundary_to_volume_ratio": "Boundary-to-volume ratio",
    "gt_elongation": "Elongation",
    "gt_n_components": "Multifocality (# components)",
    "false_positive_voxels": "False-positive voxels",
    "false_negative_voxels": "False-negative voxels",
    "boundary_error_fraction": "Boundary error fraction",
    "dice": "Dice",
}

DEPTH_LABELS = {
    "encoder1": "Enc 1",
    "encoder2": "Enc 2",
    "encoder3": "Enc 3",
    "encoder4": "Enc 4",
    "bottleneck": "Bottleneck",
    "decoder4": "Dec 4",
    "decoder3": "Dec 3",
    "decoder2": "Dec 2",
    "decoder1": "Dec 1",
}


def load_layer_index(index_path: str | Path) -> pd.DataFrame:
    """Load the layer embedding index table."""
    return pd.read_csv(index_path)


def load_layer_matrix(index_df: pd.DataFrame, layer_name: str) -> np.ndarray:
    """Stack embeddings for one layer across cases."""
    col = f"path_{layer_name}"
    if col not in index_df.columns:
        raise ValueError(f"Missing column {col} in layer index")
    return np.stack(
        [np.load(path).astype(np.float32) for path in index_df[col]],
        axis=0,
    )


def build_anatomy_table(
    index_df: pd.DataFrame,
    failure_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive anatomical and failure-morphology targets from GT masks and failure metrics."""
    merged = index_df.copy()
    if failure_df is not None:
        failure_cols = [
            c
            for c in failure_df.columns
            if c
            in {
                "case_id",
                "boundary_error_fraction",
                "false_positive_voxels",
                "false_negative_voxels",
                "dice",
            }
        ]
        merged = merged.merge(
            failure_df[failure_cols].drop_duplicates("case_id"),
            on="case_id",
            how="left",
            suffixes=("", "_fail"),
        )
        if "dice_fail" in merged.columns:
            merged["dice"] = merged["dice"].fillna(merged["dice_fail"])
            merged = merged.drop(columns=["dice_fail"])

    rows: list[dict[str, float]] = []
    for record in merged.to_dict(orient="records"):
        gt = np.load(record["path_ground_truth"]).astype(np.int16)
        wt_mask = whole_tumor_mask(gt).astype(bool)
        wt_vol = int(wt_mask.sum())
        nec = int((gt == 1).sum())
        edema = int((gt == 2).sum())
        enh = int((gt == 3).sum())

        if wt_vol > 0:
            eroded = ndimage.binary_erosion(wt_mask)
            surface = int((wt_mask & ~eroded).sum())
            compactness = surface / (wt_vol ** (2.0 / 3.0) + 1e-8)
            boundary_to_volume_ratio = surface / (wt_vol + 1e-8)
            coords = np.argwhere(wt_mask)
            centroid = coords.mean(axis=0) / np.array(gt.shape, dtype=np.float64)
            if len(coords) > 3:
                evals = np.sort(np.linalg.eigvalsh(np.cov(coords.T)))[::-1]
                elongation = float(evals[0] / (evals[-1] + 1e-8))
            else:
                elongation = 1.0
            _, n_comp = ndimage.label(wt_mask.astype(np.uint8))
        else:
            surface = 0
            compactness = 0.0
            boundary_to_volume_ratio = 0.0
            centroid = np.array([0.5, 0.5, 0.5])
            elongation = 1.0
            n_comp = 0

        rows.append(
            {
                "case_id": record["case_id"],
                "centroid_x": float(centroid[2]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[0]),
                "gt_wt_voxels": float(wt_vol),
                "log_wt_volume": float(np.log1p(wt_vol)),
                "gt_edema_voxels": float(edema),
                "gt_enhancing_voxels": float(enh),
                "gt_necrosis_voxels": float(nec),
                "gt_edema_frac": float(edema / (wt_vol + 1e-8)),
                "gt_enhancing_frac": float(enh / (wt_vol + 1e-8)),
                "gt_necrosis_frac": float(nec / (wt_vol + 1e-8)),
                "gt_compactness": float(compactness),
                "boundary_to_volume_ratio": float(boundary_to_volume_ratio),
                "gt_elongation": float(elongation),
                "gt_n_components": float(n_comp),
                "gt_surface_voxels": float(surface),
                "boundary_error_fraction": float(record.get("boundary_error_fraction", np.nan)),
                "false_positive_voxels": float(record.get("false_positive_voxels", np.nan)),
                "false_negative_voxels": float(record.get("false_negative_voxels", np.nan)),
                "dice": float(record.get("dice", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _cv_regression_metrics(
    x: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int = 42,
) -> dict[str, float]:
    """Fold-safe Ridge regression metrics."""
    y = np.asarray(y, dtype=np.float64)
    if np.std(y) < 1e-12:
        return {"r2": float("nan"), "mae": float("nan"), "spearman": float("nan")}

    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    preds = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in splitter.split(x):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-2, 3, 20))),
            ]
        )
        pipe.fit(x[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(x[test_idx])

    rho, _ = spearmanr(y, preds)
    return {
        "r2": float(r2_score(y, preds)),
        "mae": float(mean_absolute_error(y, preds)),
        "spearman": float(rho) if not np.isnan(rho) else float("nan"),
    }


def run_layer_recoverability(
    index_df: pd.DataFrame,
    anatomy_df: pd.DataFrame,
    output_path: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train fold-safe Ridge probes for each layer × anatomical target."""
    cv_folds = choose_cv_folds(np.ones(len(index_df), dtype=int))
    cv_folds = max(cv_folds, 2)

    rows: list[dict[str, float | str | int]] = []
    for layer_name in LAYER_ORDER:
        embeddings = load_layer_matrix(index_df, layer_name)
        for target_key, target_label in ANATOMY_TARGET_SPECS.items():
            y = anatomy_df[target_key].values.astype(np.float64)
            metrics = _cv_regression_metrics(
                _impute_nan(embeddings),
                y,
                cv_folds=cv_folds,
                random_state=random_state,
            )
            rows.append(
                {
                    "layer": layer_name,
                    "target": target_key,
                    "target_label": target_label,
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "spearman": metrics["spearman"],
                    "n_cases": len(index_df),
                    "cv_folds": cv_folds,
                }
            )

    result = pd.DataFrame(rows)
    ensure_dir(output_path.parent)
    result.to_csv(output_path, index=False)
    return result


def _evaluate_bad_case_classifier(
    x: np.ndarray,
    labels: np.ndarray,
    classifier_name: str,
    random_state: int = 42,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    cv_folds = choose_cv_folds(labels)
    if cv_folds < 2 or len(np.unique(labels)) < 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "bad_case_recall": float("nan"),
            "accuracy": float("nan"),
            "cv_folds": 0,
        }

    proba = np.zeros(len(labels), dtype=np.float64)
    pred_hard = np.zeros(len(labels), dtype=int)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in splitter.split(x, labels):
        x_train = _impute_nan(x[train_idx])
        x_test = _impute_nan(x[test_idx])
        if classifier_name == "logistic_regression":
            model = LogisticRegression(max_iter=3000, random_state=random_state)
        elif classifier_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200, random_state=random_state, n_jobs=1
            )
        else:
            raise ValueError(classifier_name)

        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(x_train, labels[train_idx])
        fold_proba = pipe.predict_proba(x_test)[:, 1]
        proba[test_idx] = fold_proba
        pred_hard[test_idx] = (fold_proba >= 0.5).astype(int)

    return {
        "auroc": float(roc_auc_score(labels, proba)),
        "auprc": float(average_precision_score(labels, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred_hard)),
        "f1": float(f1_score(labels, pred_hard, zero_division=0)),
        "bad_case_recall": float(recall_score(labels, pred_hard, zero_division=0)),
        "accuracy": float(accuracy_score(labels, pred_hard)),
        "cv_folds": cv_folds,
    }


def run_layer_quality_analysis(
    index_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    output_path: Path,
    dice_threshold: float = 0.80,
    bad_quantile: float = 0.25,
    use_pca: bool = False,
    pca_components: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate bad-case detection from uncertainty, each layer, and their combination."""
    merged = index_df.merge(
        failure_df.drop_duplicates("case_id"),
        on="case_id",
        how="inner",
        suffixes=("", "_fail"),
    )
    dice = merged["dice"].values.astype(np.float64)
    if "dice_fail" in merged.columns:
        dice = np.where(np.isnan(dice), merged["dice_fail"].values, dice)

    uncertainty, _ = compute_deployable_uncertainty_features(merged)

    rows: list[dict[str, float | str | int]] = []
    bad_case_modes = [
        ("threshold", dice_threshold, None),
        ("quantile", float("nan"), bad_quantile),
    ]

    for bad_mode, threshold, quantile in bad_case_modes:
        labels, effective_threshold = create_bad_case_label(
            dice,
            mode=bad_mode,
            threshold=threshold if bad_mode == "threshold" else 0.80,
            bad_quantile=quantile,
        )
        n_bad = int(labels.sum())

        for classifier in ("logistic_regression", "random_forest"):
            unc_metrics = _evaluate_bad_case_classifier(
                uncertainty, labels, classifier, random_state=random_state
            )
            rows.append(
                {
                    "layer": "uncertainty_only",
                    "feature_set": "uncertainty_only",
                    "classifier": classifier,
                    "bad_case_mode": bad_mode,
                    "dice_threshold": effective_threshold,
                    "n_cases": len(labels),
                    "n_bad_cases": n_bad,
                    **unc_metrics,
                }
            )

            for layer_name in LAYER_ORDER:
                layer_emb = load_layer_matrix(merged, layer_name)
                if use_pca:
                    layer_emb = _fold_safe_pca_features(
                        layer_emb, labels, pca_components, random_state
                    )

                layer_metrics = _evaluate_bad_case_classifier(
                    layer_emb, labels, classifier, random_state=random_state
                )
                rows.append(
                    {
                        "layer": layer_name,
                        "feature_set": "layer_only",
                        "classifier": classifier,
                        "bad_case_mode": bad_mode,
                        "dice_threshold": effective_threshold,
                        "n_cases": len(labels),
                        "n_bad_cases": n_bad,
                        **layer_metrics,
                    }
                )

                combined = np.concatenate([uncertainty, layer_emb], axis=1).astype(np.float32)
                combined_metrics = _evaluate_bad_case_classifier(
                    combined, labels, classifier, random_state=random_state
                )
                rows.append(
                    {
                        "layer": layer_name,
                        "feature_set": "uncertainty_plus_layer",
                        "classifier": classifier,
                        "bad_case_mode": bad_mode,
                        "dice_threshold": effective_threshold,
                        "n_cases": len(labels),
                        "n_bad_cases": n_bad,
                        **combined_metrics,
                    }
                )

    result = pd.DataFrame(rows)
    ensure_dir(output_path.parent)
    result.to_csv(output_path, index=False)
    return result


def _fold_safe_pca_features(
    x: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    """Out-of-fold PCA projection for optional dimensionality reduction."""
    cv_folds = choose_cv_folds(labels)
    if cv_folds < 2:
        return x

    transformed = np.zeros_like(x, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in splitter.split(x, labels):
        scaler = StandardScaler().fit(x[train_idx])
        n_comp = min(n_components, x.shape[1], len(train_idx) - 1)
        pca = PCA(n_components=n_comp, random_state=random_state)
        x_train = pca.fit_transform(scaler.transform(x[train_idx]))
        x_test = pca.transform(scaler.transform(x[test_idx]))
        transformed[test_idx, : x_test.shape[1]] = x_test
    return transformed.astype(np.float32)


def _best_layer_row(df: pd.DataFrame, metric: str, target: str) -> pd.Series:
    sub = df[(df["target"] == target)].copy()
    return sub.loc[sub[metric].idxmax()]


def _safe_idxmax(series: pd.Series, default: str = "bottleneck") -> str:
    """Return index of max value, or default when all values are NA."""
    valid = series.dropna()
    if valid.empty:
        return default
    return str(valid.idxmax())


def generate_layer_figures(
    recoverability_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Create heatmap, depth curves, AUROC bars, and uncertainty comparison plots."""
    ensure_dir(figures_dir)
    plt.rcParams.update(PUBLICATION_RC)

    plot_df = recoverability_df.copy()
    plot_df["layer_label"] = plot_df["layer"].map(DEPTH_LABELS)
    heatmap = plot_df.pivot(index="layer_label", columns="target_label", values="r2")
    heatmap = heatmap.reindex([DEPTH_LABELS[l] for l in LAYER_ORDER])

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heatmap.values, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=1.0)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_title("Anatomical recoverability by layer (CV R²)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "heatmap_layer_target_r2.png", bbox_inches="tight")
    plt.close(fig)

    depth_targets = {
        "centroid_z": "Centroid Z",
        "log_wt_volume": "log(WT volume)",
        "gt_compactness": "Boundary complexity",
        "dice": "Dice",
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(LAYER_ORDER))
    for target_key, label in depth_targets.items():
        sub = recoverability_df[recoverability_df["target"] == target_key]
        sub = sub.set_index("layer").reindex(LAYER_ORDER)
        ax.plot(x, sub["r2"].values, marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([DEPTH_LABELS[l] for l in LAYER_ORDER], rotation=45, ha="right")
    ax.set_ylabel("CV R²")
    ax.set_title("Recoverability across network depth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "line_recoverability_across_depth.png", bbox_inches="tight")
    plt.close(fig)

    qsub = quality_df[
        (quality_df["bad_case_mode"] == "threshold")
        & (quality_df["classifier"] == "logistic_regression")
        & (quality_df["feature_set"] == "layer_only")
    ].set_index("layer").reindex(LAYER_ORDER)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(LAYER_ORDER)), qsub["auroc"].values, color="steelblue")
    ax.set_xticks(np.arange(len(LAYER_ORDER)))
    ax.set_xticklabels([DEPTH_LABELS[l] for l in LAYER_ORDER], rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Bad-case detection (Dice < 0.80) by layer embedding")
    unc_rows = quality_df[
        (quality_df["layer"] == "uncertainty_only")
        & (quality_df["bad_case_mode"] == "threshold")
        & (quality_df["classifier"] == "logistic_regression")
    ]
    unc_auroc = float(unc_rows["auroc"].iloc[0]) if not unc_rows.empty else float("nan")
    if not np.isnan(unc_auroc):
        ax.axhline(
            unc_auroc,
            color="crimson",
            linestyle="--",
            label=f"uncertainty only ({unc_auroc:.3f})",
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "bar_badcase_auroc_by_layer.png", bbox_inches="tight")
    plt.close(fig)

    # Best layer vs uncertainty vs combined
    layer_aurocs = qsub["auroc"]
    best_layer = _safe_idxmax(layer_aurocs)
    best_layer_auroc = float(layer_aurocs.max()) if not layer_aurocs.dropna().empty else float("nan")
    combined_rows = quality_df[
        (quality_df["layer"] == best_layer)
        & (quality_df["feature_set"] == "uncertainty_plus_layer")
        & (quality_df["bad_case_mode"] == "threshold")
        & (quality_df["classifier"] == "logistic_regression")
    ]
    combined_auroc = float(combined_rows["auroc"].iloc[0]) if not combined_rows.empty else float("nan")
    labels = ["Uncertainty only", f"Best layer ({DEPTH_LABELS[best_layer]})", "Uncertainty + best layer"]
    values = [unc_auroc, best_layer_auroc, combined_auroc]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["crimson", "steelblue", "seagreen"])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.0)
    ax.set_title("Quality estimation comparison (Dice < 0.80)")
    fig.tight_layout()
    fig.savefig(figures_dir / "bar_uncertainty_vs_best_layer.png", bbox_inches="tight")
    plt.close(fig)


def generate_layer_report(
    recoverability_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    output_path: Path,
    n_cases: int,
) -> str:
    """Write markdown report answering the layer-distribution hypotheses."""
    def best_for(target: str, metric: str = "r2") -> tuple[str, float]:
        sub = recoverability_df[recoverability_df["target"] == target]
        row = sub.loc[sub[metric].idxmax()]
        return str(row["layer"]), float(row[metric])

    unc_rows = quality_df[
        (quality_df["layer"] == "uncertainty_only")
        & (quality_df["bad_case_mode"] == "threshold")
        & (quality_df["classifier"] == "logistic_regression")
    ]
    unc_auroc = float(unc_rows["auroc"].iloc[0]) if not unc_rows.empty else float("nan")
    q_layer = quality_df[
        (quality_df["bad_case_mode"] == "threshold")
        & (quality_df["classifier"] == "logistic_regression")
        & (quality_df["feature_set"] == "layer_only")
    ]
    best_quality_layer = _safe_idxmax(q_layer.set_index("layer")["auroc"])
    best_quality_auroc = float(
        q_layer.loc[q_layer["layer"] == best_quality_layer, "auroc"].iloc[0]
    ) if not q_layer.empty else float("nan")
    combined_rows = quality_df[
        (quality_df["layer"] == best_quality_layer)
        & (quality_df["feature_set"] == "uncertainty_plus_layer")
        & (quality_df["bad_case_mode"] == "threshold")
        & (quality_df["classifier"] == "logistic_regression")
    ]
    combined_auroc = float(combined_rows["auroc"].iloc[0]) if not combined_rows.empty else float("nan")

    loc_layer, loc_r2 = best_for("centroid_z")
    vol_layer, vol_r2 = best_for("log_wt_volume")
    bnd_layer, bnd_r2 = best_for("gt_compactness")
    dice_layer, dice_r2 = best_for("dice")
    edema_layer, edema_r2 = best_for("gt_edema_frac")
    enh_layer, enh_r2 = best_for("gt_enhancing_frac")
    nec_layer, nec_r2 = best_for("gt_necrosis_frac")
    bn_bnd = float(
        recoverability_df[
            (recoverability_df["layer"] == "bottleneck")
            & (recoverability_df["target"] == "gt_compactness")
        ]["r2"].iloc[0]
    )
    best_bnd_layer, best_bnd_r2 = bnd_layer, bnd_r2

    report = f"""# Layer Analysis Report

**Validation cases:** {n_cases}

## Research questions

### Does the bottleneck discard fine morphology?

Compare bottleneck R² for boundary complexity ({bn_bnd:.3f}) vs best layer ({DEPTH_LABELS.get(best_bnd_layer, best_bnd_layer)}: {best_bnd_r2:.3f}).
If decoder/early-encoder layers exceed the bottleneck on boundary, composition, or failure-morphology targets, the bottleneck compresses fine detail.

### Which layer best encodes tumor location?

Best layer for centroid Z: **{DEPTH_LABELS.get(loc_layer, loc_layer)}** (R² = {loc_r2:.3f}).

### Which layer best encodes tumor volume?

Best layer for log(WT volume): **{DEPTH_LABELS.get(vol_layer, vol_layer)}** (R² = {vol_r2:.3f}).

### Which layer best encodes boundary complexity?

Best layer: **{DEPTH_LABELS.get(bnd_layer, bnd_layer)}** (R² = {bnd_r2:.3f}); bottleneck R² = {bn_bnd:.3f}.

### Which layer best encodes tissue composition?

| Subregion fraction | Best layer | R² |
|--------------------|------------|-----|
| Edema | {DEPTH_LABELS.get(edema_layer, edema_layer)} | {edema_r2:.3f} |
| Enhancing | {DEPTH_LABELS.get(enh_layer, enh_layer)} | {enh_r2:.3f} |
| Necrosis | {DEPTH_LABELS.get(nec_layer, nec_layer)} | {nec_r2:.3f} |

### Which layer best predicts segmentation failure (Dice)?

Best layer for Dice regression: **{DEPTH_LABELS.get(dice_layer, dice_layer)}** (R² = {dice_r2:.3f}).

### Which layer best predicts bad cases (Dice < 0.80)?

| Method | AUROC |
|--------|-------|
| Uncertainty only | {unc_auroc:.3f} |
| Best layer embedding ({DEPTH_LABELS.get(best_quality_layer, best_quality_layer)}) | {best_quality_auroc:.3f} |
| Uncertainty + best layer | {combined_auroc:.3f} |

### Do skip/decoder features contain information missing from the bottleneck?

See `layer_recoverability.csv` heatmap (`figures/heatmap_layer_target_r2.png`).
Targets where decoder or encoder layers beat bottleneck R² indicate distributed anatomical coding.

### Does any layer improve over uncertainty for quality estimation?

Primary hypothesis is **not** that embeddings beat uncertainty. On this run, uncertainty AUROC = {unc_auroc:.3f}; best layer = {best_quality_auroc:.3f}; combined = {combined_auroc:.3f}.

## Artifacts

- `layer_recoverability.csv`
- `layer_quality_results.csv`
- `figures/heatmap_layer_target_r2.png`
- `figures/line_recoverability_across_depth.png`
- `figures/bar_badcase_auroc_by_layer.png`
- `figures/bar_uncertainty_vs_best_layer.png`
"""
    ensure_dir(output_path.parent)
    output_path.write_text(report)
    return report


def analyze_layers(
    layer_index_path: str | Path,
    failure_table_path: str | Path,
    output_dir: str | Path,
    dice_threshold: float = 0.80,
    bad_quantile: float = 0.25,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Run full multi-layer recoverability and quality analysis."""
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    index_df = load_layer_index(layer_index_path)
    failure_df = pd.read_csv(failure_table_path)
    anatomy_df = build_anatomy_table(index_df, failure_df)

    recoverability_df = run_layer_recoverability(
        index_df,
        anatomy_df,
        output_dir / "layer_recoverability.csv",
        random_state=random_state,
    )
    quality_df = run_layer_quality_analysis(
        index_df,
        failure_df,
        output_dir / "layer_quality_results.csv",
        dice_threshold=dice_threshold,
        bad_quantile=bad_quantile,
        random_state=random_state,
    )

    generate_layer_figures(recoverability_df, quality_df, figures_dir)
    generate_layer_report(
        recoverability_df,
        quality_df,
        output_dir / "layer_analysis_report.md",
        n_cases=len(index_df),
    )

    return {
        "recoverability": recoverability_df,
        "quality": quality_df,
        "anatomy": anatomy_df,
    }
