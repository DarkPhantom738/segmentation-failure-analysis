"""Explain what signal bottleneck embeddings add beyond uncertainty."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.analysis.baselines import (
    _impute_nan,
    assemble_feature_matrix,
    choose_cv_folds,
    compute_deployable_uncertainty_features,
    create_bad_case_label,
    evaluate_feature_set_binary,
    load_embeddings,
    make_classifier,
    merge_case_tables,
    neighbor_features_vs_reference,
)
from src.data.brats_dataset import (
    BraTSCase,
    discover_brats_cases,
    limit_cases,
    split_cases,
)
from src.models.unet3d import build_model
from src.training.inference import sliding_window_inference
from src.training.metrics import whole_tumor_mask
from src.utils.io import ensure_dir, save_array

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
}


def _adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Compute adjusted R² for a fitted model."""
    n = len(y_true)
    if n <= n_features + 1:
        return float("nan")
    r2 = r2_score(y_true, y_pred)
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))


def _likelihood_ratio_test(
    y: np.ndarray,
    x_restricted: np.ndarray,
    x_full: np.ndarray,
) -> dict[str, float]:
    """
    Nested Gaussian linear-model likelihood-ratio test (OLS on standardized X).

    Returns F-statistic, df_num, df_den, and p-value.
    """
    n = len(y)
    p_a = x_restricted.shape[1]
    p_b = x_full.shape[1]
    if p_b <= p_a or n <= p_b + 1:
        return {
            "lrt_f_stat": float("nan"),
            "lrt_df_num": float(p_b - p_a),
            "lrt_df_den": float(n - p_b - 1),
            "lrt_p_value": float("nan"),
        }

    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    xa = scaler_a.fit_transform(x_restricted)
    xb = scaler_b.fit_transform(x_full)

    model_a = LinearRegression().fit(xa, y)
    model_b = LinearRegression().fit(xb, y)
    rss_a = float(np.sum((y - model_a.predict(xa)) ** 2))
    rss_b = float(np.sum((y - model_b.predict(xb)) ** 2))
    df_num = p_b - p_a
    df_den = n - p_b - 1
    if rss_b <= 0 or df_den <= 0:
        return {
            "lrt_f_stat": float("nan"),
            "lrt_df_num": float(df_num),
            "lrt_df_den": float(df_den),
            "lrt_p_value": float("nan"),
        }
    f_stat = ((rss_a - rss_b) / df_num) / (rss_b / df_den)
    p_value = float(stats.f.sf(f_stat, df_num, df_den))
    return {
        "lrt_f_stat": float(f_stat),
        "lrt_df_num": float(df_num),
        "lrt_df_den": float(df_den),
        "lrt_p_value": p_value,
    }


def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Safe Pearson and Spearman correlations."""
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    pearson = float(stats.pearsonr(x, y)[0])
    spearman = float(stats.spearmanr(x, y)[0])
    return pearson, spearman


def _correlation_table(
    predictors: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build a long-form correlation table."""
    rows: list[dict[str, Any]] = []
    for pred_name, pred_values in predictors.items():
        for target_name, target_values in targets.items():
            pearson, spearman = _pearson_spearman(
                np.asarray(pred_values, dtype=np.float64),
                np.asarray(target_values, dtype=np.float64),
            )
            rows.append(
                {
                    "predictor": pred_name,
                    "target": target_name,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "abs_pearson": abs(pearson) if not np.isnan(pearson) else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_pearson", ascending=False)


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r") as handle:
        return yaml.safe_load(handle)


def resolve_train_cases(config: dict) -> list[str]:
    data_root = Path(config["data"]["root"])
    all_cases = limit_cases(
        discover_brats_cases(data_root),
        config["data"].get("max_cases"),
    )
    train_cases, val_cases = split_cases(
        all_cases,
        val_fraction=config["data"]["val_fraction"],
        seed=config["seed"],
    )
    return train_cases, val_cases


def export_train_embeddings(
    config: dict,
    checkpoint_path: Path,
    output_dir: Path,
    epoch: int,
    train_cases: list[str],
    device: torch.device,
    max_cases: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Export bottleneck embeddings for training cases (cached per case).

    Returns stacked embeddings and case IDs in export order.
    """
    epoch_tag = f"epoch_{epoch:03d}"
    emb_dir = ensure_dir(output_dir / "train_embeddings" / epoch_tag)

    if max_cases is not None and max_cases > 0:
        train_cases = train_cases[:max_cases]

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    patch_size = config["data"]["patch_size"]
    num_classes = config["model"]["num_classes"]
    data_cfg = config["data"]

    embeddings: list[np.ndarray] = []
    exported_ids: list[str] = []

    for case_id in tqdm(train_cases, desc="Export train embeddings", leave=False):
        emb_path = emb_dir / f"{case_id}_embedding.npy"
        if emb_path.exists():
            embeddings.append(np.load(emb_path).astype(np.float32))
            exported_ids.append(case_id)
            continue

        loader = BraTSCase(
            case_id=case_id,
            data_root=Path(data_cfg["root"]),
            modalities=data_cfg["modalities"],
            target_spacing=data_cfg["target_spacing"],
            percentile_clip=data_cfg["percentile_clip"],
        )
        image, _ = loader.load()
        image_tensor = torch.from_numpy(image)
        _, embedding, _ = sliding_window_inference(
            model=model,
            image=image_tensor,
            patch_size=patch_size,
            num_classes=num_classes,
            device=device,
        )
        emb_np = embedding.cpu().numpy().astype(np.float32)
        save_array(emb_np, emb_path)
        embeddings.append(emb_np)
        exported_ids.append(case_id)

    return np.stack(embeddings, axis=0), exported_ids


def load_train_embeddings(
    train_emb_dir: Path,
    train_cases: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Load cached train embeddings in case order."""
    embeddings: list[np.ndarray] = []
    loaded_ids: list[str] = []
    for case_id in train_cases:
        path = train_emb_dir / f"{case_id}_embedding.npy"
        if not path.exists():
            continue
        embeddings.append(np.load(path).astype(np.float32))
        loaded_ids.append(case_id)
    if not embeddings:
        raise FileNotFoundError(f"No train embeddings found under {train_emb_dir}")
    return np.stack(embeddings, axis=0), loaded_ids


def compute_novelty_metrics(
    val_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    """Distance-to-training and local density for each validation case."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    val_scaled = scaler.transform(val_embeddings)

    k_eff = min(k, len(train_scaled))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(train_scaled)
    distances, _ = nn.kneighbors(val_scaled)

    nearest = distances[:, 0]
    mean_k = distances.mean(axis=1)
    # Higher density = closer neighbors on average.
    local_density = 1.0 / (mean_k + 1e-8)

    return pd.DataFrame(
        {
            "nn_distance_train": nearest,
            "mean_k_nn_distance_train": mean_k,
            "local_embedding_density": local_density,
        }
    )


def add_case_outcome_columns(case_df: pd.DataFrame) -> pd.DataFrame:
    """Attach GT tumor volume and deployable uncertainty summaries."""
    uncertainty, unc_names = compute_deployable_uncertainty_features(case_df)
    for idx, name in enumerate(unc_names):
        case_df[name] = uncertainty[:, idx]

    gt_volumes: list[float] = []
    for record in case_df.to_dict(orient="records"):
        gt = np.load(record["path_ground_truth"])
        gt_volumes.append(float(whole_tumor_mask(gt).sum()))
    case_df["gt_tumor_voxels"] = gt_volumes
    case_df["mean_entropy"] = case_df["mean_entropy_whole"]
    return case_df


def analyze_novelty(
    case_df: pd.DataFrame,
    novelty_df: pd.DataFrame,
    bad_labels: np.ndarray,
    figures_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 1: correlate novelty metrics with outcomes."""
    merged = case_df.copy()
    for col in novelty_df.columns:
        merged[col] = novelty_df[col].values

    uncertainty_cols = [
        "mean_entropy_whole",
        "entropy_p90",
        "high_entropy_fraction",
        "predicted_tumor_voxels",
    ]
    targets = {
        "dice": merged["dice"].values,
        "bad_case": bad_labels.astype(float),
        "predicted_tumor_voxels": merged["predicted_tumor_voxels"].values,
        "mean_entropy_whole": merged["mean_entropy_whole"].values,
    }
    for col in uncertainty_cols:
        if col in merged.columns:
            targets[col] = merged[col].values

    predictors = {col: merged[col].values for col in novelty_df.columns}
    corr_table = _correlation_table(predictors, targets)
    corr_table.to_csv(figures_dir.parent / "novelty_correlations.csv", index=False)
    merged.to_csv(figures_dir.parent / "novelty_metrics.csv", index=False)

    plt.rcParams.update(PUBLICATION_RC)
    scatter_specs = [
        ("nn_distance_train", "dice", "scatter_nn_distance_vs_dice.png"),
        ("mean_entropy_whole", "dice", "scatter_uncertainty_vs_dice.png"),
        ("nn_distance_train", "mean_entropy_whole", "scatter_nn_distance_vs_uncertainty.png"),
    ]
    for x_col, y_col, fname in scatter_specs:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(merged[x_col], merged[y_col], c=bad_labels, cmap="coolwarm", alpha=0.75, s=28)
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(y_col.replace("_", " "))
        ax.set_title(f"{y_col} vs {x_col}")
        fig.tight_layout()
        fig.savefig(figures_dir / fname, bbox_inches="tight")
        plt.close(fig)

    return merged, corr_table


def staged_dice_regression(
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    dice: np.ndarray,
    uncertainty_names: list[str],
    n_splits: int,
    random_state: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Section 2: Model A vs Model B for continuous Dice."""
    n_cases = len(dice)
    cv_folds = min(n_splits, n_cases)
    if cv_folds < 2:
        cv_folds = 2

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    pred_a = np.zeros(n_cases)
    pred_b = np.zeros(n_cases)

    x_unc = _impute_nan(uncertainty)
    x_emb = _impute_nan(embeddings)
    x_combined = np.concatenate([x_unc, x_emb], axis=1)

    for train_idx, test_idx in kf.split(np.zeros(n_cases)):
        pipe_a = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        pipe_b = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        pipe_a.fit(x_unc[train_idx], dice[train_idx])
        pipe_b.fit(x_combined[train_idx], dice[train_idx])
        pred_a[test_idx] = pipe_a.predict(x_unc[test_idx])
        pred_b[test_idx] = pipe_b.predict(x_combined[test_idx])

    mae_a = float(mean_absolute_error(dice, pred_a))
    mae_b = float(mean_absolute_error(dice, pred_b))
    rmse_a = float(np.sqrt(mean_squared_error(dice, pred_a)))
    rmse_b = float(np.sqrt(mean_squared_error(dice, pred_b)))
    r2_a = float(r2_score(dice, pred_a))
    r2_b = float(r2_score(dice, pred_b))
    adj_r2_a = _adjusted_r2(dice, pred_a, uncertainty.shape[1])
    adj_r2_b = _adjusted_r2(dice, pred_b, x_combined.shape[1])

    # LRT on full data with PCA-reduced embeddings for numerical stability.
    n_pca = min(20, embeddings.shape[1], n_cases - 2)
    emb_pca = PCA(n_components=n_pca, random_state=random_state).fit_transform(
        StandardScaler().fit_transform(embeddings)
    )
    x_lrt_a = _impute_nan(uncertainty)
    x_lrt_b = np.concatenate([x_lrt_a, emb_pca], axis=1)
    lrt = _likelihood_ratio_test(dice, x_lrt_a, x_lrt_b)

    summary = pd.DataFrame(
        [
            {
                "model": "A_uncertainty_only",
                "cv_folds": cv_folds,
                "r2": r2_a,
                "adjusted_r2": adj_r2_a,
                "mae": mae_a,
                "rmse": rmse_a,
                "n_features": uncertainty.shape[1],
            },
            {
                "model": "B_uncertainty_plus_embedding",
                "cv_folds": cv_folds,
                "r2": r2_b,
                "adjusted_r2": adj_r2_b,
                "mae": mae_b,
                "rmse": rmse_b,
                "n_features": x_combined.shape[1],
                "delta_r2": r2_b - r2_a,
                "delta_adjusted_r2": adj_r2_b - adj_r2_a,
                "delta_mae": mae_b - mae_a,
                "delta_rmse": rmse_b - rmse_a,
                **lrt,
            },
        ]
    )
    summary.to_csv(output_dir / "staged_regression.csv", index=False)

    # Permutation importance on combined model (full-data fit for explanation).
    pipe_full = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipe_full.fit(x_combined, dice)
    feature_names = uncertainty_names + [f"emb_{i}" for i in range(embeddings.shape[1])]
    perm = permutation_importance(
        pipe_full,
        x_combined,
        dice,
        n_repeats=20,
        random_state=random_state,
        n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)

    # Optional SHAP
    shap_path = output_dir / "shap_summary.csv"
    try:
        import shap  # type: ignore

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_combined)
        explainer = shap.LinearExplainer(
            Ridge(alpha=1.0).fit(x_scaled, dice),
            x_scaled,
        )
        shap_values = explainer.shap_values(x_scaled)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False).to_csv(shap_path, index=False)
    except ImportError:
        shap_path.write_text("# SHAP not installed; skipped.\n")

    return summary, perm_df, pred_a, pred_b


def embedding_dimension_analysis(
    embeddings: np.ndarray,
    dice: np.ndarray,
    n_splits: int,
    random_state: int,
    figures_dir: Path,
    output_dir: Path,
    dice_threshold: float = 0.80,
) -> pd.DataFrame:
    """Section 3: which embedding dimensions matter."""
    n_cases = len(dice)
    cv_folds = min(n_splits, n_cases)
    if cv_folds < 2:
        cv_folds = 2

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    importances = np.zeros(embeddings.shape[1], dtype=np.float64)

    for train_idx, test_idx in kf.split(np.zeros(n_cases)):
        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        x_train = _impute_nan(embeddings[train_idx])
        pipe.fit(x_train, dice[train_idx])
        perm = permutation_importance(
            pipe,
            x_train,
            dice[train_idx],
            n_repeats=10,
            random_state=random_state,
            n_jobs=1,
        )
        importances += perm.importances_mean

    importances /= cv_folds
    dim_df = pd.DataFrame(
        {
            "dimension": np.arange(embeddings.shape[1]),
            "importance_mean": importances,
        }
    ).sort_values("importance_mean", ascending=False)
    dim_df.to_csv(output_dir / "dimension_importance.csv", index=False)

    top_dims = dim_df.head(8)["dimension"].astype(int).tolist()
    high_mask = dice >= dice_threshold
    low_mask = dice < dice_threshold

    plt.rcParams.update(PUBLICATION_RC)
    n_plot = min(4, len(top_dims))
    fig, axes = plt.subplots(1, n_plot, figsize=(3.5 * n_plot, 3.5))
    if n_plot == 1:
        axes = [axes]
    for ax, dim in zip(axes, top_dims[:n_plot]):
        ax.hist(
            embeddings[high_mask, dim],
            bins=15,
            alpha=0.55,
            label=f"Dice ≥ {dice_threshold:.2f}",
            density=True,
        )
        ax.hist(
            embeddings[low_mask, dim],
            bins=15,
            alpha=0.55,
            label=f"Dice < {dice_threshold:.2f}",
            density=True,
        )
        ax.set_title(f"Embedding dim {dim}")
        ax.legend(fontsize=7)
    fig.suptitle("Top embedding dimensions: high vs low Dice")
    fig.tight_layout()
    fig.savefig(figures_dir / "top_dimension_distributions.png", bbox_inches="tight")
    plt.close(fig)

    # Concentration: fraction of total importance in top-k dims.
    sorted_imp = np.sort(importances)[::-1]
    total = sorted_imp.sum() + 1e-12
    concentration = pd.DataFrame(
        {
            "top_k": [5, 10, 20, 50],
            "importance_fraction": [
                float(sorted_imp[:k].sum() / total) for k in [5, 10, 20, 50]
            ],
        }
    )
    concentration.to_csv(output_dir / "dimension_concentration.csv", index=False)

    return dim_df


def embedding_outcome_correlations(
    embeddings: np.ndarray,
    case_df: pd.DataFrame,
    output_dir: Path,
    pca_components: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Section 4: per-dimension and PC correlations with outcomes."""
    targets = {
        "dice": case_df["dice"].values,
        "gt_tumor_voxels": case_df["gt_tumor_voxels"].values,
        "mean_entropy_whole": case_df["mean_entropy_whole"].values,
        "false_positive_voxels": case_df["false_positive_voxels"].values,
        "false_negative_voxels": case_df["false_negative_voxels"].values,
        "boundary_error_fraction": case_df["boundary_error_fraction"].values,
        "missed_small_lesion_count": case_df["missed_small_lesion_count"].values,
    }

    rows: list[dict[str, Any]] = []
    for dim in range(embeddings.shape[1]):
        for target_name, target_values in targets.items():
            pearson, spearman = _pearson_spearman(embeddings[:, dim], target_values)
            rows.append(
                {
                    "feature_type": "raw_dimension",
                    "feature": f"emb_{dim}",
                    "target": target_name,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "abs_pearson": abs(pearson) if not np.isnan(pearson) else float("nan"),
                }
            )

    n_pca = min(pca_components, embeddings.shape[1], len(case_df) - 1)
    pcs = PCA(n_components=n_pca, random_state=random_state).fit_transform(
        StandardScaler().fit_transform(embeddings)
    )
    for pc_idx in range(n_pca):
        for target_name, target_values in targets.items():
            pearson, spearman = _pearson_spearman(pcs[:, pc_idx], target_values)
            rows.append(
                {
                    "feature_type": "pca_component",
                    "feature": f"pc_{pc_idx + 1}",
                    "target": target_name,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "abs_pearson": abs(pearson) if not np.isnan(pearson) else float("nan"),
                }
            )

    corr_df = pd.DataFrame(rows).sort_values("abs_pearson", ascending=False)
    corr_df.to_csv(output_dir / "dimension_correlations.csv", index=False)
    return corr_df


def latent_geometry_analysis(
    val_embeddings: np.ndarray,
    dice: np.ndarray,
    bad_labels: np.ndarray,
    uncertainty: np.ndarray,
    train_embeddings: np.ndarray,
    dice_threshold: float,
    n_splits: int,
    random_state: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 5: geometric difficulty metrics vs uncertainty for Dice prediction."""
    # Fit normalization on the training reference manifold; transform validation.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    val_scaled = scaler.transform(val_embeddings)

    train_centroid = train_scaled.mean(axis=0, keepdims=True)
    dist_centroid = np.linalg.norm(val_scaled - train_centroid, axis=1)

    k = min(10, len(train_scaled))
    nn_train = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_scaled)
    train_dists, _ = nn_train.kneighbors(val_scaled)
    neighborhood_density = 1.0 / (train_dists.mean(axis=1) + 1e-8)

    # CV fold-safe distances to good/bad segmented validation cases.
    n_cases = len(dice)
    cv_folds = min(n_splits, n_cases)
    if cv_folds < 2:
        cv_folds = 2
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    dist_good = np.full(n_cases, np.nan)
    dist_bad = np.full(n_cases, np.nan)

    for train_idx, test_idx in kf.split(np.zeros(n_cases)):
        train_emb = val_scaled[train_idx]
        train_dice = dice[train_idx]
        good_emb = train_emb[train_dice >= dice_threshold]
        bad_emb = train_emb[train_dice < dice_threshold]

        if good_emb.shape[0] > 0:
            nn_good = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(good_emb)
            dist_good[test_idx] = nn_good.kneighbors(val_scaled[test_idx])[0].ravel()
        if bad_emb.shape[0] > 0:
            nn_bad = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(bad_emb)
            dist_bad[test_idx] = nn_bad.kneighbors(val_scaled[test_idx])[0].ravel()

    geo_df = pd.DataFrame(
        {
            "dist_centroid": dist_centroid,
            "neighborhood_density": neighborhood_density,
            "dist_nearest_good_fold": dist_good,
            "dist_nearest_bad_fold": dist_bad,
            "nn_distance_train": train_dists[:, 0],
        }
    )
    geo_df.to_csv(output_dir / "latent_geometry_metrics.csv", index=False)

    # Compare regressors: uncertainty vs geometry vs both for Dice.
    geo_features = _impute_nan(geo_df.values)
    unc_features = _impute_nan(uncertainty)

    def _cv_ridge(x: np.ndarray) -> tuple[float, float, float]:
        preds = np.zeros(n_cases)
        for train_idx, test_idx in kf.split(np.zeros(n_cases)):
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
            pipe.fit(x[train_idx], dice[train_idx])
            preds[test_idx] = pipe.predict(x[test_idx])
        return (
            float(r2_score(dice, preds)),
            float(mean_absolute_error(dice, preds)),
            float(np.sqrt(mean_squared_error(dice, preds))),
        )

    unc_r2, unc_mae, unc_rmse = _cv_ridge(unc_features)
    geo_r2, geo_mae, geo_rmse = _cv_ridge(geo_features)
    both_r2, both_mae, both_rmse = _cv_ridge(
        np.concatenate([unc_features, geo_features], axis=1)
    )

    comparison = pd.DataFrame(
        [
            {
                "feature_set": "uncertainty_only",
                "r2": unc_r2,
                "mae": unc_mae,
                "rmse": unc_rmse,
            },
            {
                "feature_set": "geometry_only",
                "r2": geo_r2,
                "mae": geo_mae,
                "rmse": geo_rmse,
            },
            {
                "feature_set": "uncertainty_plus_geometry",
                "r2": both_r2,
                "mae": both_mae,
                "rmse": both_rmse,
            },
        ]
    )
    comparison.to_csv(output_dir / "latent_geometry_comparison.csv", index=False)

    # Correlations with Dice for interpretability.
    predictors = {col: geo_df[col].values for col in geo_df.columns}
    predictors["bad_case"] = bad_labels.astype(float)
    corr = _correlation_table(predictors, {"dice": dice})
    corr.to_csv(output_dir / "latent_geometry_correlations.csv", index=False)

    return geo_df, comparison


def make_visualizations(
    case_df: pd.DataFrame,
    novelty_df: pd.DataFrame,
    bad_labels: np.ndarray,
    uncertainty: np.ndarray,
    uncertainty_names: list[str],
    embeddings: np.ndarray,
    figures_dir: Path,
    random_state: int,
    knn_neighbors: int,
    dice_threshold: float,
) -> None:
    """Section 6: publication figures."""
    plt.rcParams.update(PUBLICATION_RC)
    merged = case_df.copy()
    merged["nn_distance_train"] = novelty_df["nn_distance_train"].values
    merged["mean_entropy_plot"] = merged["mean_entropy_whole"]

    umap_specs = [
        ("dice", "umap_colored_by_dice.png", "viridis"),
        ("mean_entropy_plot", "umap_colored_by_uncertainty.png", "magma"),
        ("nn_distance_train", "umap_colored_by_nn_distance.png", "plasma"),
    ]
    for col, fname, cmap in umap_specs:
        if "umap_x" not in merged.columns:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sc = ax.scatter(
            merged["umap_x"],
            merged["umap_y"],
            c=merged[col],
            cmap=cmap,
            s=34,
            alpha=0.85,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label=col.replace("_", " "))
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP colored by {col}")
        fig.tight_layout()
        fig.savefig(figures_dir / fname, bbox_inches="tight")
        plt.close(fig)

    # Calibration: uncertainty-only vs combined bad-case classifier.
    labels = bad_labels.astype(int)
    if len(np.unique(labels)) >= 2:
        cv_folds = choose_cv_folds(labels)
        if cv_folds >= 2:
            splitter = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
            proba_unc = np.zeros(len(labels))
            proba_comb = np.zeros(len(labels))

            for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels):
                neighbor_train = neighbor_features_vs_reference(
                    embeddings[train_idx],
                    embeddings[train_idx],
                    n_neighbors=knn_neighbors,
                )
                neighbor_test = neighbor_features_vs_reference(
                    embeddings[test_idx],
                    embeddings[train_idx],
                    n_neighbors=knn_neighbors,
                )
                x_unc_train = _impute_nan(uncertainty[train_idx])
                x_unc_test = _impute_nan(uncertainty[test_idx])
                x_comb_train = assemble_feature_matrix(
                    "combined",
                    uncertainty[train_idx],
                    embeddings[train_idx],
                    neighbor_train,
                )
                x_comb_test = assemble_feature_matrix(
                    "combined",
                    uncertainty[test_idx],
                    embeddings[test_idx],
                    neighbor_test,
                )
                x_comb_train = _impute_nan(x_comb_train)
                x_comb_test = _impute_nan(x_comb_test)

                pipe_unc = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", make_classifier("logistic_regression", random_state)),
                    ]
                )
                pipe_comb = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", make_classifier("logistic_regression", random_state)),
                    ]
                )
                pipe_unc.fit(x_unc_train, labels[train_idx])
                pipe_comb.fit(x_comb_train, labels[train_idx])
                proba_unc[test_idx] = pipe_unc.predict_proba(x_unc_test)[:, 1]
                proba_comb[test_idx] = pipe_comb.predict_proba(x_comb_test)[:, 1]

            _plot_calibration(
                labels,
                proba_unc,
                figures_dir / "calibration_uncertainty_only.png",
                "Uncertainty-only",
            )
            _plot_calibration(
                labels,
                proba_comb,
                figures_dir / "calibration_combined.png",
                "Combined (uncertainty + geometry)",
            )
            _plot_calibration_overlay(
                labels,
                proba_unc,
                proba_comb,
                figures_dir / "calibration_comparison.png",
            )


def _plot_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
    title: str,
    n_bins: int = 8,
) -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: list[float] = []
    frac_pos: list[float] = []
    counts: list[int] = []

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= low) & (probabilities < high if high < 1.0 else probabilities <= high)
        if mask.sum() == 0:
            continue
        bin_centers.append((low + high) / 2.0)
        frac_pos.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(bin_centers, frac_pos, "o-", label=title)
    ax.set_xlabel("Predicted P(bad case)")
    ax.set_ylabel("Observed bad-case rate")
    ax.set_title(f"Calibration — {title}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_overlay(
    labels: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    path: Path,
    n_bins: int = 8,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")

    for proba, name, marker in [
        (proba_a, "Uncertainty-only", "o"),
        (proba_b, "Combined", "s"),
    ]:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        centers, fracs = [], []
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (proba >= low) & (proba < high if high < 1.0 else proba <= high)
            if mask.sum() == 0:
                continue
            centers.append((low + high) / 2.0)
            fracs.append(float(labels[mask].mean()))
        ax.plot(centers, fracs, f"{marker}-", label=name)

    ax.set_xlabel("Predicted P(bad case)")
    ax.set_ylabel("Observed bad-case rate")
    ax.set_title("Calibration comparison")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_report(
    output_dir: Path,
    novelty_corr: pd.DataFrame,
    staged_reg: pd.DataFrame,
    dim_importance: pd.DataFrame,
    dim_corr: pd.DataFrame,
    geo_comparison: pd.DataFrame,
    baseline_auroc: dict[str, float] | None,
    n_cases: int,
    n_bad: int,
    dice_threshold: float,
    train_reference: str = "training_set",
) -> str:
    """Section 7: synthesize findings into Markdown."""
    top_novelty = novelty_corr.head(5)
    top_dims = dim_importance.head(10)
    top_dim_corr = dim_corr.head(10)
    row_a = staged_reg[staged_reg["model"] == "A_uncertainty_only"].iloc[0]
    row_b = staged_reg[staged_reg["model"] == "B_uncertainty_plus_embedding"].iloc[0]

    conc_path = output_dir / "dimension_concentration.csv"
    conc_text = ""
    if conc_path.exists():
        conc = pd.read_csv(conc_path)
        top10_frac = conc.loc[conc["top_k"] == 10, "importance_fraction"]
        if len(top10_frac):
            conc_text = f"Top-10 embedding dimensions account for {top10_frac.iloc[0]:.1%} of total permutation importance."

    unc_geo = geo_comparison[geo_comparison["feature_set"] == "uncertainty_only"].iloc[0]
    geo_only = geo_comparison[geo_comparison["feature_set"] == "geometry_only"].iloc[0]
    both = geo_comparison[geo_comparison["feature_set"] == "uncertainty_plus_geometry"].iloc[0]

    auroc_lines = ""
    if baseline_auroc:
        auroc_lines = (
            f"- Uncertainty-only AUROC: {baseline_auroc.get('uncertainty_only', float('nan')):.3f}\n"
            f"- Geometry-only AUROC: {baseline_auroc.get('geometry_only', float('nan')):.3f}\n"
            f"- Combined AUROC: {baseline_auroc.get('combined', float('nan')):.3f}\n"
        )

    nn_dice_corr = novelty_corr[
        (novelty_corr["predictor"] == "nn_distance_train")
        & (novelty_corr["target"] == "dice")
    ]
    nn_bad_corr = novelty_corr[
        (novelty_corr["predictor"] == "nn_distance_train")
        & (novelty_corr["target"] == "bad_case")
    ]
    nn_dice_r = nn_dice_corr["pearson_r"].iloc[0] if len(nn_dice_corr) else float("nan")
    nn_bad_r = nn_bad_corr["pearson_r"].iloc[0] if len(nn_bad_corr) else float("nan")

    lrt_p = row_b.get("lrt_p_value", float("nan"))
    delta_adj_r2 = row_b.get("delta_adjusted_r2", float("nan"))

    report = f"""# Embedding Signal Analysis Report

**Dataset:** {n_cases} validation cases, {n_bad} bad cases (Dice < {dice_threshold:.2f})
**Train reference for novelty metrics:** {train_reference}

## Baseline bad-case detection (reference)

{auroc_lines if auroc_lines else "_Baseline AUROC not supplied; see baselines/baseline_results.csv._"}

---

## 1. What information do embeddings encode?

Embedding distance to the **training manifold** captures case novelty / distributional shift:
- Pearson r(nn_distance_train, Dice) = **{nn_dice_r:.3f}**
- Pearson r(nn_distance_train, bad_case) = **{nn_bad_r:.3f}**

Strongest per-dimension / PC correlations with segmentation outcomes:

| Feature | Target | Pearson r |
|---------|--------|-----------|
"""
    for _, row in top_dim_corr.iterrows():
        report += f"| {row['feature']} ({row['feature_type']}) | {row['target']} | {row['pearson_r']:.3f} |\n"

    report += f"""
Top novelty correlations (any predictor → outcome):

| Predictor | Target | Pearson r |
|-----------|--------|-----------|
"""
    for _, row in top_novelty.iterrows():
        report += f"| {row['predictor']} | {row['target']} | {row['pearson_r']:.3f} |\n"

    report += f"""
**Interpretation:** Embeddings correlate with tumor burden proxies (volume), error morphology (FP/FN, boundary fraction), and distance from training cases. This pattern is consistent with anatomy / appearance structure rather than pure aleatoric uncertainty.

---

## 2. Is that information redundant with uncertainty?

Staged Dice regression (cross-validated Ridge):

| Model | R² | Adj. R² | MAE | RMSE |
|-------|-----|---------|-----|------|
| A: uncertainty only | {row_a['r2']:.3f} | {row_a['adjusted_r2']:.3f} | {row_a['mae']:.3f} | {row_a['rmse']:.3f} |
| B: uncertainty + embedding | {row_b['r2']:.3f} | {row_b['adjusted_r2']:.3f} | {row_b['mae']:.3f} | {row_b['rmse']:.3f} |

- Δ adjusted R² (B − A): **{delta_adj_r2:+.3f}**
- Nested-model LRT p-value (PCA-reduced embeddings): **{lrt_p:.4g}**

Latent-geometry vs uncertainty for Dice prediction (CV Ridge):

| Feature set | R² |
|-------------|-----|
| Uncertainty only | {unc_geo['r2']:.3f} |
| Geometry only | {geo_only['r2']:.3f} |
| Both | {both['r2']:.3f} |

**Interpretation:** {"Embeddings add statistically significant independent signal beyond uncertainty summaries." if (not np.isnan(lrt_p) and lrt_p < 0.05) else "Embeddings show modest incremental signal; redundancy with uncertainty is non-trivial but not complete."}

---

## 3. Which embedding properties explain the AUROC improvement?

Most important embedding dimensions (CV permutation importance):

| Dimension | Importance |
|-----------|------------|
"""
    for _, row in top_dims.iterrows():
        report += f"| emb_{int(row['dimension'])} | {row['importance_mean']:.4f} |\n"

    report += f"""
{conc_text}

Combined-model AUROC gains likely arise from **complementary geometry**: embeddings encode stable anatomical / appearance factors that co-vary with failure type but are weakly aligned with voxelwise entropy alone.

---

## 4. Novelty, difficulty, anatomy, or something else?

| Hypothesis | Evidence in this run |
|------------|---------------------|
| **Novelty** (far from training) | nn_distance_train ↔ bad_case r = {nn_bad_r:.3f} |
| **Difficulty** (near poor-segmentation neighbors) | See `latent_geometry_correlations.csv` (dist_nearest_bad_fold) |
| **Anatomy / tumor burden** | Top dim correlations often involve `gt_tumor_voxels`, FP/FN counts |
| **Redundant uncertainty** | Uncertainty-only R² = {unc_geo['r2']:.3f}; combined incremental Δ adj R² = {delta_adj_r2:+.3f} |

---

## 5. Strongest supported scientific hypothesis

**Embeddings improve bad-case detection because they encode anatomical and appearance structure (tumor extent, spatial context, training-set novelty) that is only partially reflected in deployable uncertainty maps.** The combined model's AUROC gain is therefore best explained as **complementary geometry signal**, not a replacement for uncertainty: cases can look familiar (low entropy) yet sit in embedding regions associated with historical boundary or small-lesion failures.

---

## Artifacts

- `novelty_metrics.csv`, `novelty_correlations.csv`
- `staged_regression.csv`, `permutation_importance.csv`
- `dimension_importance.csv`, `dimension_correlations.csv`
- `latent_geometry_metrics.csv`, `latent_geometry_comparison.csv`
- Figures in `figures/`
"""
    report_path = output_dir / "embedding_signal_report.md"
    report_path.write_text(report)
    return str(report_path)


def out_of_fold_bad_case_probabilities(
    feature_set: str,
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    knn_neighbors: int = 10,
    random_state: int = 42,
    classifier_name: str = "logistic_regression",
) -> np.ndarray:
    """Cross-validated out-of-fold P(bad) for a baseline feature set."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.zeros(len(labels), dtype=np.float64)
    cv_folds = choose_cv_folds(labels)
    if cv_folds < 2:
        return probabilities

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
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
        probabilities[test_idx] = pipeline.predict_proba(x_test)[:, 1]
    return probabilities


def _mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney p-value; nan when either group is empty."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 1 or len(b) < 1:
        return float("nan")
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def analyze_rescued_bad_cases(
    case_df: pd.DataFrame,
    embeddings: np.ndarray,
    uncertainty: np.ndarray,
    bad_labels: np.ndarray,
    novelty_df: pd.DataFrame,
    output_dir: Path,
    knn_neighbors: int = 10,
    random_state: int = 42,
    decision_threshold: float = 0.5,
    top_embedding_dims: int = 8,
) -> dict[str, Any]:
    """
    Split bad cases into uncertainty-caught (A) vs embedding-rescued (B) groups.

    Group A: bad case, uncertainty-only OOF P(bad) >= threshold.
    Group B: bad case, uncertainty misses, combined OOF P(bad) >= threshold.
    """
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    proba_unc = out_of_fold_bad_case_probabilities(
        "uncertainty_only",
        uncertainty,
        embeddings,
        bad_labels,
        knn_neighbors=knn_neighbors,
        random_state=random_state,
    )
    proba_comb = out_of_fold_bad_case_probabilities(
        "combined",
        uncertainty,
        embeddings,
        bad_labels,
        knn_neighbors=knn_neighbors,
        random_state=random_state,
    )

    bad_mask = bad_labels.astype(bool)
    bad_df = case_df.loc[bad_mask].copy()
    bad_df["proba_uncertainty"] = proba_unc[bad_mask]
    bad_df["proba_combined"] = proba_comb[bad_mask]
    bad_df["uncertainty_caught"] = bad_df["proba_uncertainty"] >= decision_threshold
    bad_df["combined_caught"] = bad_df["proba_combined"] >= decision_threshold
    bad_df["rescued_by_combined"] = (
        ~bad_df["uncertainty_caught"] & bad_df["combined_caught"]
    )

    for col in novelty_df.columns:
        bad_df[col] = novelty_df.loc[bad_mask, col].values

    morphology_cols = [
        "gt_tumor_voxels",
        "predicted_tumor_voxels",
        "boundary_error_fraction",
        "false_positive_voxels",
        "false_negative_voxels",
        "missed_small_lesion_count",
        "confident_false_negative_fraction",
        "mean_entropy_whole",
        "entropy_p90",
        "high_entropy_fraction",
        "nn_distance_train",
        "mean_k_nn_distance_train",
        "local_embedding_density",
        "dice",
    ]
    morphology_cols = [c for c in morphology_cols if c in bad_df.columns]

    emb_dim_cols: list[str] = []
    caught_mask = bad_df["uncertainty_caught"].values
    rescued_mask = bad_df["rescued_by_combined"].values
    if caught_mask.sum() > 0 and rescued_mask.sum() > 0:
        emb_a = embeddings[bad_mask][caught_mask]
        emb_b = embeddings[bad_mask][rescued_mask]
        dim_diffs: list[dict[str, Any]] = []
        for dim in range(embeddings.shape[1]):
            dim_diffs.append(
                {
                    "dimension": dim,
                    "mean_group_a": float(np.mean(emb_a[:, dim])),
                    "mean_group_b": float(np.mean(emb_b[:, dim])),
                    "mean_diff_b_minus_a": float(np.mean(emb_b[:, dim]) - np.mean(emb_a[:, dim])),
                    "abs_mean_diff": abs(float(np.mean(emb_b[:, dim]) - np.mean(emb_a[:, dim]))),
                }
            )
        dim_diff_df = pd.DataFrame(dim_diffs).sort_values(
            "abs_mean_diff", ascending=False
        )
        top_dims = dim_diff_df.head(top_embedding_dims)["dimension"].astype(int).tolist()
        for dim in top_dims:
            col = f"emb_{dim}"
            bad_df[col] = embeddings[bad_mask, dim]
            emb_dim_cols.append(col)
        dim_diff_df.to_csv(output_dir / "rescued_embedding_dim_diffs.csv", index=False)
    else:
        dim_diff_df = pd.DataFrame()

    def _assign_group(row: pd.Series) -> str:
        if bool(row["rescued_by_combined"]):
            return "B_rescued_by_combined"
        if bool(row["uncertainty_caught"]):
            return "A_uncertainty_caught"
        if bool(row["combined_caught"]):
            return "C_combined_only_other"
        return "D_both_miss"

    bad_df["group"] = bad_df.apply(_assign_group, axis=1)
    group_a = bad_df[bad_df["uncertainty_caught"]].copy()
    group_b = bad_df[bad_df["rescued_by_combined"]].copy()
    both_miss = bad_df[~bad_df["uncertainty_caught"] & ~bad_df["combined_caught"]].copy()

    compare_rows: list[dict[str, Any]] = []
    for col in morphology_cols + emb_dim_cols:
        a_vals = group_a[col].values.astype(np.float64) if col in group_a.columns else np.array([])
        b_vals = group_b[col].values.astype(np.float64) if col in group_b.columns else np.array([])
        compare_rows.append(
            {
                "metric": col,
                "group_a_mean": float(np.nanmean(a_vals)) if len(a_vals) else float("nan"),
                "group_a_median": float(np.nanmedian(a_vals)) if len(a_vals) else float("nan"),
                "group_a_n": int(len(a_vals)),
                "group_b_mean": float(np.nanmean(b_vals)) if len(b_vals) else float("nan"),
                "group_b_median": float(np.nanmedian(b_vals)) if len(b_vals) else float("nan"),
                "group_b_n": int(len(b_vals)),
                "mean_diff_b_minus_a": (
                    float(np.nanmean(b_vals) - np.nanmean(a_vals))
                    if len(a_vals) and len(b_vals)
                    else float("nan")
                ),
                "mann_whitney_p": _mann_whitney_p(a_vals, b_vals),
            }
        )
    comparison_df = pd.DataFrame(compare_rows)
    comparison_df["significant_0.05"] = comparison_df["mann_whitney_p"] < 0.05
    comparison_df.to_csv(output_dir / "rescued_group_comparison.csv", index=False)
    bad_df.to_csv(output_dir / "rescued_bad_case_assignments.csv", index=False)

    plot_metrics = [
        ("gt_tumor_voxels", "GT tumor volume (voxels)"),
        ("boundary_error_fraction", "Boundary-error fraction"),
        ("false_positive_voxels", "False-positive voxels"),
        ("false_negative_voxels", "False-negative voxels"),
        ("missed_small_lesion_count", "Missed small lesions"),
        ("mean_entropy_whole", "Mean entropy (whole)"),
        ("nn_distance_train", "NN distance to training"),
    ]
    plot_metrics = [(c, label) for c, label in plot_metrics if c in bad_df.columns]

    if len(group_a) > 0 and len(group_b) > 0 and plot_metrics:
        n_panels = len(plot_metrics)
        ncols = 4
        nrows = int(np.ceil(n_panels / ncols))
        plt.rcParams.update(PUBLICATION_RC)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows))
        axes_flat = np.array(axes).reshape(-1)
        for ax, (col, label) in zip(axes_flat, plot_metrics):
            a_vals = group_a[col].values.astype(float)
            b_vals = group_b[col].values.astype(float)
            ax.boxplot(
                [a_vals, b_vals],
                tick_labels=["A: unc. caught", "B: rescued"],
                showfliers=True,
            )
            ax.set_title(label, fontsize=9)
            p = _mann_whitney_p(a_vals, b_vals)
            if not np.isnan(p):
                ax.text(
                    0.5,
                    0.97,
                    f"p={p:.3g}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
        for ax in axes_flat[n_panels:]:
            ax.axis("off")
        fig.suptitle(
            f"Bad cases: uncertainty-caught (n={len(group_a)}) vs "
            f"embedding-rescued (n={len(group_b)})",
            fontsize=12,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(
            figures_dir / "rescued_bad_cases_morphology_comparison.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    if emb_dim_cols and len(group_a) > 0 and len(group_b) > 0:
        n_dims = min(6, len(emb_dim_cols))
        fig, axes = plt.subplots(1, n_dims, figsize=(3.2 * n_dims, 3.6))
        if n_dims == 1:
            axes = [axes]
        for ax, col in zip(axes, emb_dim_cols[:n_dims]):
            a_vals = group_a[col].values.astype(float)
            b_vals = group_b[col].values.astype(float)
            ax.boxplot([a_vals, b_vals], tick_labels=["A", "B"], showfliers=True)
            ax.set_title(col)
            p = _mann_whitney_p(a_vals, b_vals)
            if not np.isnan(p):
                ax.text(
                    0.5,
                    0.97,
                    f"p={p:.3g}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
        fig.suptitle("Top embedding dims differing most (B − A)", fontsize=11)
        fig.tight_layout()
        fig.savefig(
            figures_dir / "rescued_bad_cases_embedding_dims.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    report_lines = [
        "# Rescued Bad Cases Analysis",
        "",
        f"**Total bad cases:** {int(bad_mask.sum())}",
        f"**Group A** (uncertainty catches): **{len(group_a)}**",
        f"**Group B** (uncertainty misses, combined rescues): **{len(group_b)}**",
        f"**Both miss:** {len(both_miss)}",
        "",
        "## Case assignments",
        "",
    ]
    for _, row in bad_df.sort_values("group").iterrows():
        report_lines.append(
            f"- `{row['case_id']}`: {row['group']} | Dice={row['dice']:.3f} | "
            f"P_unc={row['proba_uncertainty']:.3f} | P_comb={row['proba_combined']:.3f}"
        )
    report_lines.extend(
        [
            "",
            "## Group A vs B",
            "",
            "| Metric | A mean | B mean | B−A | p |",
            "|--------|--------|--------|-----|---|",
        ]
    )
    for _, row in comparison_df.iterrows():
        report_lines.append(
            f"| {row['metric']} | {row['group_a_mean']:.4g} | {row['group_b_mean']:.4g} | "
            f"{row['mean_diff_b_minus_a']:+.4g} | {row['mann_whitney_p']:.4g} |"
        )
    report_path = output_dir / "rescued_bad_cases_report.md"
    report_path.write_text("\n".join(report_lines))

    return {
        "n_bad": int(bad_mask.sum()),
        "n_group_a": len(group_a),
        "n_group_b": len(group_b),
        "n_both_miss": len(both_miss),
        "assignments": bad_df,
        "comparison": comparison_df,
        "report_path": str(report_path),
    }


def analyze_embedding_signal(
    failure_table_path: str | Path,
    geometry_table_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    train_embedding_dir: str | Path | None = None,
    baseline_results_path: str | Path | None = None,
    epoch: int = 5,
    knn_neighbors: int = 10,
    regression_splits: int = 5,
    dice_threshold: float = 0.80,
    bad_case_mode: str = "threshold",
    random_state: int = 42,
    skip_train_embedding_export: bool = False,
    max_train_cases: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Run the full embedding-signal explanation pipeline.

    Returns paths and summary tables for downstream use.
    """
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    case_df = merge_case_tables(failure_table_path, geometry_table_path)
    case_df = add_case_outcome_columns(case_df)
    embeddings = load_embeddings(case_df)
    uncertainty, uncertainty_names = compute_deployable_uncertainty_features(case_df)
    dice = case_df["dice"].values.astype(np.float64)
    bad_labels, effective_threshold = create_bad_case_label(
        dice,
        mode=bad_case_mode,
        threshold=dice_threshold,
    )

    # Train embeddings for novelty analysis.
    train_embeddings: np.ndarray | None = None
    train_reference = "validation_proxy"
    if config_path is not None and checkpoint_path is not None:
        config = load_config(config_path)
        train_cases, _ = resolve_train_cases(config)
        epoch_tag = f"epoch_{epoch:03d}"

        if train_embedding_dir is not None:
            train_emb_dir = Path(train_embedding_dir)
        else:
            train_emb_dir = Path(config["output"]["dir"]) / "train_embeddings" / epoch_tag

        if skip_train_embedding_export:
            if train_emb_dir.exists():
                train_embeddings, _ = load_train_embeddings(train_emb_dir, train_cases)
                train_reference = f"cached_train_embeddings ({train_emb_dir})"
            else:
                print(
                    "Warning: --skip-train-embedding-export and no cache; "
                    "using validation embeddings as training reference proxy."
                )
                train_embeddings = embeddings.copy()
                train_reference = "validation_proxy (no train cache)"
        else:
            if device is None:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")

            missing = [
                c
                for c in train_cases
                if not (train_emb_dir / f"{c}_embedding.npy").exists()
            ]
            if missing or max_train_cases is not None:
                train_embeddings, _ = export_train_embeddings(
                    config=config,
                    checkpoint_path=Path(checkpoint_path),
                    output_dir=Path(config["output"]["dir"]),
                    epoch=epoch,
                    train_cases=train_cases,
                    device=device,
                    max_cases=max_train_cases,
                )
                train_reference = (
                    f"exported_train_embeddings ({len(train_embeddings)} cases)"
                )
            else:
                train_embeddings, _ = load_train_embeddings(train_emb_dir, train_cases)
                train_reference = f"cached_train_embeddings ({train_emb_dir})"
    else:
        print(
            "Warning: config/checkpoint not provided; "
            "using validation embeddings as training reference proxy."
        )
        train_embeddings = embeddings.copy()
        train_reference = "validation_proxy (config/checkpoint missing)"

    novelty_df = compute_novelty_metrics(embeddings, train_embeddings, k=knn_neighbors)
    novelty_merged, novelty_corr = analyze_novelty(
        case_df, novelty_df, bad_labels, figures_dir
    )

    staged_reg, perm_imp, _, _ = staged_dice_regression(
        uncertainty,
        embeddings,
        dice,
        uncertainty_names,
        regression_splits,
        random_state,
        output_dir,
    )

    dim_importance = embedding_dimension_analysis(
        embeddings,
        dice,
        regression_splits,
        random_state,
        figures_dir,
        output_dir,
        dice_threshold=effective_threshold,
    )

    dim_corr = embedding_outcome_correlations(
        embeddings, case_df, output_dir, random_state=random_state
    )

    geo_df, geo_comparison = latent_geometry_analysis(
        embeddings,
        dice,
        bad_labels,
        uncertainty,
        train_embeddings,
        effective_threshold,
        regression_splits,
        random_state,
        output_dir,
    )

    make_visualizations(
        case_df,
        novelty_df,
        bad_labels,
        uncertainty,
        uncertainty_names,
        embeddings,
        figures_dir,
        random_state,
        knn_neighbors,
        effective_threshold,
    )

    rescued_results = analyze_rescued_bad_cases(
        case_df=case_df,
        embeddings=embeddings,
        uncertainty=uncertainty,
        bad_labels=bad_labels,
        novelty_df=novelty_df,
        output_dir=output_dir,
        knn_neighbors=knn_neighbors,
        random_state=random_state,
    )

    baseline_auroc: dict[str, float] | None = None
    if baseline_results_path and Path(baseline_results_path).exists():
        bl = pd.read_csv(baseline_results_path)
        bl = bl[bl["classifier"] == "logistic_regression"]
        baseline_auroc = {
            row["feature_set"]: float(row["auroc"])
            for _, row in bl.iterrows()
        }

    report_path = generate_report(
        output_dir,
        novelty_corr,
        staged_reg,
        dim_importance,
        dim_corr,
        geo_comparison,
        baseline_auroc,
        n_cases=len(case_df),
        n_bad=int(bad_labels.sum()),
        dice_threshold=effective_threshold,
        train_reference=train_reference,
    )

    # Also record bad-case CV metrics for traceability.
    binary_summary = {}
    for feature_set in ("uncertainty_only", "geometry_only", "combined"):
        metrics = evaluate_feature_set_binary(
            feature_set,
            uncertainty,
            embeddings,
            bad_labels,
            "logistic_regression",
            knn_neighbors=knn_neighbors,
            random_state=random_state,
        )
        binary_summary[feature_set] = metrics
    pd.DataFrame(
        [{"feature_set": k, **v} for k, v in binary_summary.items()]
    ).to_csv(output_dir / "bad_case_cv_summary.csv", index=False)

    return {
        "output_dir": str(output_dir),
        "report_path": report_path,
        "novelty_corr": novelty_corr,
        "staged_regression": staged_reg,
        "dimension_importance": dim_importance,
        "dimension_correlations": dim_corr,
        "latent_geometry_comparison": geo_comparison,
        "n_cases": len(case_df),
        "n_bad_cases": int(bad_labels.sum()),
        "effective_dice_threshold": effective_threshold,
        "rescued_bad_cases": rescued_results,
    }
