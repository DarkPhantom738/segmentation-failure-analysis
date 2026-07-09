"""Baseline comparisons for failure detection, failure-type prediction, and Dice regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.analysis.geometry import load_embeddings
from src.training.metrics import whole_tumor_mask
from src.utils.io import ensure_dir

FEATURE_SETS = (
    "uncertainty_only",
    "geometry_only",
    "combined",
)

CLASSIFIERS = (
    "logistic_regression",
    "random_forest",
)

REGRESSORS = (
    "ridge",
    "random_forest",
)


def create_bad_case_label(
    dice: np.ndarray,
    mode: str = "threshold",
    threshold: float = 0.80,
    bad_quantile: float | None = None,
    bottom_percentile: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Binary label for poor segmentation performance.

    Modes:
      - threshold: bad if dice < fixed threshold (e.g. 0.80 or 0.85)
      - quantile: bad if dice < empirical quantile of this set
        (e.g. bad_quantile=0.25 -> bottom 25%)
      - bottom_percentile: legacy alias for quantile
        (bottom_percentile=25 -> quantile 0.25)

    Returns:
      labels, effective_threshold used to define bad cases
    """
    dice = np.asarray(dice, dtype=np.float64)

    if mode == "bottom_percentile":
        mode = "quantile"
        if bad_quantile is None:
            pct = 25.0 if bottom_percentile is None else float(bottom_percentile)
            bad_quantile = pct / 100.0

    if mode == "threshold":
        effective_threshold = float(threshold)
    elif mode == "quantile":
        if bad_quantile is None:
            bad_quantile = 0.25
        if not 0.0 < float(bad_quantile) < 1.0:
            raise ValueError(f"bad_quantile must be in (0, 1), got {bad_quantile}")
        effective_threshold = float(np.quantile(dice, float(bad_quantile)))
    else:
        raise ValueError(
            f"Unknown bad-case mode: {mode}. Use 'threshold' or 'quantile'."
        )

    labels = (dice < effective_threshold).astype(int)
    # If quantile ties leave zero bad cases, mark the lowest dice as bad.
    if labels.sum() == 0 and len(dice) > 0:
        labels[int(np.argmin(dice))] = 1
        effective_threshold = float(dice[labels.astype(bool)].max())
    return labels, effective_threshold


def _resolve_probability_path(record: dict) -> Path | None:
    """Find a softmax probability map path without requiring GT."""
    for key in ("path_tta_probability", "path_probability"):
        value = record.get(key)
        if isinstance(value, str) and value and Path(value).exists():
            return Path(value)

    pred_path = record.get("path_prediction")
    if isinstance(pred_path, str) and pred_path:
        candidate = Path(
            pred_path.replace("/predictions/", "/probabilities_tta/").replace(
                "_pred_tta.npy", "_probs_tta.npy"
            )
        )
        if candidate.exists():
            return candidate
        candidate = Path(
            pred_path.replace("/predictions/", "/probabilities/").replace(
                "_pred.npy", "_probs.npy"
            )
        )
        if candidate.exists():
            return candidate
    return None


def compute_deployable_uncertainty_features(
    case_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Build uncertainty / prediction-summary features that do NOT require ground truth.

    Allowed: entropy maps, predicted masks, softmax probabilities.
    Forbidden as inputs: mean_entropy_error, entropy_error_auroc/auprc,
    uncertainty-error overlap, or any feature computed from the GT error map.
    """
    feature_names = [
        "mean_entropy_whole",
        "mean_entropy_nonzero",
        "mean_entropy_pred_tumor",
        "max_entropy",
        "entropy_p50",
        "entropy_p75",
        "entropy_p90",
        "entropy_p95",
        "high_entropy_fraction",
        "predicted_tumor_fraction",
        "predicted_tumor_voxels",
        "mean_max_softmax_whole",
        "mean_max_softmax_pred_tumor",
        "mean_prob_entropy_whole",
    ]
    rows: list[list[float]] = []

    for record in case_df.to_dict(orient="records"):
        entropy = np.load(record["path_entropy"]).astype(np.float32)
        prediction = np.load(record["path_prediction"])
        pred_tumor = whole_tumor_mask(prediction).astype(bool)
        # Approximate brain/nonzero support from non-zero entropy variation / pred mask.
        nonzero = entropy > 1e-8
        if not nonzero.any():
            nonzero = np.ones_like(entropy, dtype=bool)

        flat = entropy.ravel()
        high_threshold = float(np.percentile(flat, 75))

        mean_entropy_pred_tumor = (
            float(entropy[pred_tumor].mean()) if pred_tumor.any() else float("nan")
        )

        mean_max_softmax_whole = float("nan")
        mean_max_softmax_pred_tumor = float("nan")
        mean_prob_entropy_whole = float("nan")

        prob_path = _resolve_probability_path(record)
        if prob_path is not None:
            probs = np.load(prob_path).astype(np.float32)
            # Expected shape: (C, D, H, W)
            if probs.ndim == 4:
                max_soft = probs.max(axis=0)
                mean_max_softmax_whole = float(max_soft.mean())
                if pred_tumor.any():
                    mean_max_softmax_pred_tumor = float(max_soft[pred_tumor].mean())
                # Softmax entropy: -sum p log p over classes.
                prob_entropy = -(probs * np.log(probs + 1e-8)).sum(axis=0)
                mean_prob_entropy_whole = float(prob_entropy.mean())

        rows.append(
            [
                float(entropy.mean()),
                float(entropy[nonzero].mean()),
                mean_entropy_pred_tumor,
                float(entropy.max()),
                float(np.percentile(flat, 50)),
                float(np.percentile(flat, 75)),
                float(np.percentile(flat, 90)),
                float(np.percentile(flat, 95)),
                float((flat >= high_threshold).mean()),
                float(pred_tumor.mean()),
                float(pred_tumor.sum()),
                mean_max_softmax_whole,
                mean_max_softmax_pred_tumor,
                mean_prob_entropy_whole,
            ]
        )

    return np.asarray(rows, dtype=np.float32), feature_names


def neighbor_features_vs_reference(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    n_neighbors: int = 10,
) -> np.ndarray:
    """
    Neighbor features for query points against a reference set only.

    Used inside CV folds so test cases never define the neighbor graph.
    Features: mean distance, std distance, nearest distance, similarity proxy.
    """
    n_query = query_embeddings.shape[0]
    n_ref = reference_embeddings.shape[0]
    if n_query == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if n_ref == 0:
        return np.zeros((n_query, 4), dtype=np.float32)

    scaler = StandardScaler()
    reference = scaler.fit_transform(reference_embeddings)
    query = scaler.transform(query_embeddings)

    k = min(n_neighbors, n_ref)
    # Request k+1 in case a query point is also in the reference set (self match).
    model = NearestNeighbors(n_neighbors=min(k + 1, n_ref), metric="euclidean")
    model.fit(reference)
    distances, _ = model.kneighbors(query)

    features = np.zeros((n_query, 4), dtype=np.float32)
    for i in range(n_query):
        dist_row = distances[i]
        if dist_row[0] < 1e-8 and len(dist_row) > 1:
            dist_row = dist_row[1 : k + 1]
        else:
            dist_row = dist_row[:k]
        if dist_row.size == 0:
            continue
        features[i, 0] = float(dist_row.mean())
        features[i, 1] = float(dist_row.std()) if dist_row.size > 1 else 0.0
        features[i, 2] = float(dist_row[0])
        features[i, 3] = float(1.0 / (1.0 + dist_row.mean()))
    return features


def merge_case_tables(
    failure_table_path: str | Path,
    geometry_table_path: str | Path,
) -> pd.DataFrame:
    """Merge Milestone 3 and Milestone 4 per-case tables on case_id."""
    failure_df = pd.read_csv(failure_table_path)
    geometry_df = pd.read_csv(geometry_table_path)

    geometry_cols = [
        column
        for column in ["case_id", "dominant_failure_label", "umap_x", "umap_y"]
        if column in geometry_df.columns
    ]
    return failure_df.merge(geometry_df[geometry_cols], on="case_id", how="inner")


def assemble_feature_matrix(
    feature_set: str,
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    neighbor_features: np.ndarray,
) -> np.ndarray:
    """Assemble a feature matrix for one baseline feature set."""
    if feature_set == "uncertainty_only":
        return uncertainty.astype(np.float32)
    if feature_set == "geometry_only":
        return np.concatenate([embeddings, neighbor_features], axis=1).astype(np.float32)
    if feature_set == "combined":
        return np.concatenate(
            [uncertainty, embeddings, neighbor_features], axis=1
        ).astype(np.float32)
    raise ValueError(f"Unknown feature set: {feature_set}")


def make_classifier(name: str, random_state: int = 42):
    """Create a simple sklearn classifier."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unknown classifier: {name}")


def choose_cv_folds(labels: np.ndarray, max_folds: int = 5) -> int:
    """
    Choose a stratified CV fold count that is valid for the label distribution.

    Returns 0 when there is not enough data for stratified CV
    (any class with fewer than 2 samples).
    """
    n_cases = len(labels)
    if n_cases < 4:
        return 0

    _, class_counts = np.unique(labels, return_counts=True)
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        return 0

    return min(max_folds, n_cases, min_class_count)


def _impute_nan(features: np.ndarray) -> np.ndarray:
    """Replace NaNs with column means (0 if a column is all-NaN)."""
    cleaned = features.astype(np.float64).copy()
    for col in range(cleaned.shape[1]):
        column = cleaned[:, col]
        if np.isnan(column).all():
            cleaned[:, col] = 0.0
        elif np.isnan(column).any():
            column[np.isnan(column)] = np.nanmean(column)
            cleaned[:, col] = column
    return cleaned.astype(np.float32)


def _empty_binary_metrics() -> dict[str, float]:
    return {
        "auroc": float("nan"),
        "auprc": float("nan"),
        "accuracy": float("nan"),
        "f1": float("nan"),
        "cv_folds": 0.0,
    }


def _empty_multiclass_metrics() -> dict[str, float]:
    return {
        "accuracy": float("nan"),
        "f1_macro": float("nan"),
        "cv_folds": 0.0,
    }


def evaluate_feature_set_binary(
    feature_set: str,
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifier_name: str,
    knn_neighbors: int = 10,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Cross-validated binary evaluation with fold-safe neighbor features.

    Neighbor geometry is always fit on the training fold only, so test cases
    do not define the kNN graph (avoids transductive leakage).
    """
    if len(np.unique(labels)) < 2:
        return _empty_binary_metrics()

    cv_folds = choose_cv_folds(labels)
    if cv_folds < 2:
        return _empty_binary_metrics()

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    probabilities = np.zeros(len(labels), dtype=np.float64)
    predictions = np.zeros(len(labels), dtype=np.int64)

    for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels):
        neighbor_train = neighbor_features_vs_reference(
            embeddings[train_idx], embeddings[train_idx], n_neighbors=knn_neighbors
        )
        neighbor_test = neighbor_features_vs_reference(
            embeddings[test_idx], embeddings[train_idx], n_neighbors=knn_neighbors
        )

        x_train = assemble_feature_matrix(
            feature_set, uncertainty[train_idx], embeddings[train_idx], neighbor_train
        )
        x_test = assemble_feature_matrix(
            feature_set, uncertainty[test_idx], embeddings[test_idx], neighbor_test
        )
        x_train = _impute_nan(x_train)
        x_test = _impute_nan(x_test)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", make_classifier(classifier_name, random_state=random_state)),
            ]
        )
        pipeline.fit(x_train, labels[train_idx])
        proba = pipeline.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        probabilities[test_idx] = proba
        predictions[test_idx] = pred

    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "cv_folds": float(cv_folds),
    }


def evaluate_feature_set_multiclass(
    feature_set: str,
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifier_name: str,
    knn_neighbors: int = 10,
    random_state: int = 42,
) -> dict[str, float]:
    """Cross-validated multiclass evaluation with fold-safe neighbor features."""
    if len(np.unique(labels)) < 2 or len(labels) < 2:
        return _empty_multiclass_metrics()

    cv_folds = choose_cv_folds(labels)
    if cv_folds < 2:
        return _empty_multiclass_metrics()

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    predictions = np.zeros(len(labels), dtype=np.int64)

    for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels):
        neighbor_train = neighbor_features_vs_reference(
            embeddings[train_idx], embeddings[train_idx], n_neighbors=knn_neighbors
        )
        neighbor_test = neighbor_features_vs_reference(
            embeddings[test_idx], embeddings[train_idx], n_neighbors=knn_neighbors
        )
        x_train = assemble_feature_matrix(
            feature_set, uncertainty[train_idx], embeddings[train_idx], neighbor_train
        )
        x_test = assemble_feature_matrix(
            feature_set, uncertainty[test_idx], embeddings[test_idx], neighbor_test
        )
        x_train = _impute_nan(x_train)
        x_test = _impute_nan(x_test)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", make_classifier(classifier_name, random_state=random_state)),
            ]
        )
        pipeline.fit(x_train, labels[train_idx])
        predictions[test_idx] = pipeline.predict(x_test)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "cv_folds": float(cv_folds),
    }


def run_bad_case_baselines(
    case_df: pd.DataFrame,
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    bad_case_mode: str = "threshold",
    dice_threshold: float = 0.80,
    bad_quantile: float = 0.25,
    knn_neighbors: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare uncertainty, geometry, and combined models for bad-case detection."""
    labels, effective_threshold = create_bad_case_label(
        case_df["dice"].to_numpy(),
        mode=bad_case_mode,
        threshold=dice_threshold,
        bad_quantile=bad_quantile,
    )
    rows: list[dict[str, float | str | int]] = []

    for feature_set in FEATURE_SETS:
        for classifier_name in CLASSIFIERS:
            metrics = evaluate_feature_set_binary(
                feature_set=feature_set,
                uncertainty=uncertainty,
                embeddings=embeddings,
                labels=labels,
                classifier_name=classifier_name,
                knn_neighbors=knn_neighbors,
                random_state=random_state,
            )
            rows.append(
                {
                    "task": "bad_case_detection",
                    "feature_set": feature_set,
                    "classifier": classifier_name,
                    "bad_case_mode": bad_case_mode,
                    "dice_threshold": effective_threshold,
                    "bad_quantile": bad_quantile if bad_case_mode == "quantile" else "",
                    "n_cases": len(case_df),
                    "n_bad_cases": int(labels.sum()),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def make_regressor(name: str, random_state: int = 42):
    """Create a simple sklearn regressor."""
    if name == "ridge":
        # Stronger regularization helps when embedding dim >> n_train.
        return Ridge(alpha=10.0, random_state=random_state)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unknown regressor: {name}")


def _build_regression_pipeline(
    regressor_name: str,
    n_train: int,
    n_features: int,
    random_state: int = 42,
) -> Pipeline:
    """
    Variance filter + scaler (+ optional fold-local PCA) + regressor.

    Near-constant features are dropped before scaling to avoid PCA blow-ups when
    a tiny train-fold std amplifies out-of-fold values.
    """
    steps: list[tuple] = [
        ("variance", VarianceThreshold(threshold=1e-8)),
        ("scaler", StandardScaler()),
    ]
    # When features outnumber training samples, reduce with PCA fit on train fold only.
    # Cap components conservatively for small n.
    max_components = max(1, min(n_train - 1, n_features, 8))
    if n_features >= max(8, n_train // 2) and max_components >= 1:
        steps.append(
            ("pca", PCA(n_components=max_components, random_state=random_state))
        )
    steps.append(("model", make_regressor(regressor_name, random_state=random_state)))
    return Pipeline(steps)


def evaluate_feature_set_regression(
    feature_set: str,
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    dice: np.ndarray,
    case_ids: list[str],
    regressor_name: str,
    knn_neighbors: int = 10,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Cross-validated continuous Dice prediction with fold-safe neighbor features.

    Neighbor geometry / scalers / PCA / model fitting use training folds only.
    """
    dice = np.asarray(dice, dtype=np.float64)
    n_cases = len(dice)
    if n_cases < 4:
        empty = {
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "cv_folds": 0.0,
        }
        return empty, pd.DataFrame()

    n_splits = max(2, min(n_splits, n_cases))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    predictions = np.zeros(n_cases, dtype=np.float64)

    for train_idx, test_idx in splitter.split(np.zeros(n_cases)):
        neighbor_train = neighbor_features_vs_reference(
            embeddings[train_idx], embeddings[train_idx], n_neighbors=knn_neighbors
        )
        neighbor_test = neighbor_features_vs_reference(
            embeddings[test_idx], embeddings[train_idx], n_neighbors=knn_neighbors
        )
        x_train = assemble_feature_matrix(
            feature_set, uncertainty[train_idx], embeddings[train_idx], neighbor_train
        )
        x_test = assemble_feature_matrix(
            feature_set, uncertainty[test_idx], embeddings[test_idx], neighbor_test
        )
        x_train = _impute_nan(x_train)
        x_test = _impute_nan(x_test)

        pipeline = _build_regression_pipeline(
            regressor_name=regressor_name,
            n_train=len(train_idx),
            n_features=x_train.shape[1],
            random_state=random_state,
        )
        pipeline.fit(x_train, dice[train_idx])
        predictions[test_idx] = pipeline.predict(x_test)

    pearson_r = float(pearsonr(dice, predictions)[0])
    spearman_r = float(spearmanr(dice, predictions)[0])
    mae = float(mean_absolute_error(dice, predictions))
    rmse = float(np.sqrt(mean_squared_error(dice, predictions)))
    r2 = float(r2_score(dice, predictions))

    metrics = {
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_folds": float(n_splits),
    }
    pred_df = pd.DataFrame(
        {
            "case_id": case_ids,
            "feature_set": feature_set,
            "regressor": regressor_name,
            "true_dice": dice,
            "predicted_dice": predictions,
            "absolute_error": np.abs(dice - predictions),
        }
    )
    return metrics, pred_df


def run_dice_regression(
    case_df: pd.DataFrame,
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    knn_neighbors: int = 10,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare feature sets for continuous Dice regression."""
    dice = case_df["dice"].to_numpy(dtype=np.float64)
    case_ids = case_df["case_id"].astype(str).tolist()
    metric_rows: list[dict[str, float | str | int]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_set in FEATURE_SETS:
        for regressor_name in REGRESSORS:
            metrics, pred_df = evaluate_feature_set_regression(
                feature_set=feature_set,
                uncertainty=uncertainty,
                embeddings=embeddings,
                dice=dice,
                case_ids=case_ids,
                regressor_name=regressor_name,
                knn_neighbors=knn_neighbors,
                n_splits=n_splits,
                random_state=random_state,
            )
            metric_rows.append(
                {
                    "task": "dice_regression",
                    "feature_set": feature_set,
                    "regressor": regressor_name,
                    "n_cases": len(case_df),
                    **metrics,
                }
            )
            prediction_frames.append(pred_df)

    results = pd.DataFrame(metric_rows)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return results, predictions


def run_failure_type_baselines(
    case_df: pd.DataFrame,
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    knn_neighbors: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare feature sets for dominant failure-label prediction."""
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(case_df["dominant_failure_label"].to_numpy())
    rows: list[dict[str, float | str | int]] = []

    for feature_set in FEATURE_SETS:
        for classifier_name in CLASSIFIERS:
            metrics = evaluate_feature_set_multiclass(
                feature_set=feature_set,
                uncertainty=uncertainty,
                embeddings=embeddings,
                labels=labels,
                classifier_name=classifier_name,
                knn_neighbors=knn_neighbors,
                random_state=random_state,
            )
            rows.append(
                {
                    "task": "failure_type_prediction",
                    "feature_set": feature_set,
                    "classifier": classifier_name,
                    "n_cases": len(case_df),
                    "n_failure_types": len(label_encoder.classes_),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def compare_baselines(
    failure_table_path: str | Path,
    geometry_table_path: str | Path,
    output_dir: str | Path,
    bad_case_mode: str = "threshold",
    dice_threshold: float = 0.80,
    bad_quantile: float = 0.25,
    knn_neighbors: int = 10,
    regression_splits: int = 5,
    run_regression: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run Milestone 5 baseline comparisons.

    Uses GT-free uncertainty features and fold-safe embedding neighbor features.
    Optionally also runs continuous Dice regression.
    """
    output_dir = ensure_dir(output_dir)
    case_df = merge_case_tables(failure_table_path, geometry_table_path)

    embeddings = load_embeddings(case_df)
    uncertainty, _ = compute_deployable_uncertainty_features(case_df)

    baseline_results = run_bad_case_baselines(
        case_df=case_df,
        uncertainty=uncertainty,
        embeddings=embeddings,
        bad_case_mode=bad_case_mode,
        dice_threshold=dice_threshold,
        bad_quantile=bad_quantile,
        knn_neighbors=knn_neighbors,
        random_state=random_state,
    )
    failure_type_results = run_failure_type_baselines(
        case_df=case_df,
        uncertainty=uncertainty,
        embeddings=embeddings,
        knn_neighbors=knn_neighbors,
        random_state=random_state,
    )

    if run_regression:
        regression_results, regression_predictions = run_dice_regression(
            case_df=case_df,
            uncertainty=uncertainty,
            embeddings=embeddings,
            knn_neighbors=knn_neighbors,
            n_splits=regression_splits,
            random_state=random_state,
        )
    else:
        regression_results = pd.DataFrame()
        regression_predictions = pd.DataFrame()

    baseline_results.to_csv(output_dir / "baseline_results.csv", index=False)
    failure_type_results.to_csv(output_dir / "failure_type_results.csv", index=False)
    if not regression_results.empty:
        regression_results.to_csv(output_dir / "dice_regression_results.csv", index=False)
        regression_predictions.to_csv(
            output_dir / "dice_regression_predictions.csv", index=False
        )

    return (
        baseline_results,
        failure_type_results,
        regression_results,
        regression_predictions,
    )
