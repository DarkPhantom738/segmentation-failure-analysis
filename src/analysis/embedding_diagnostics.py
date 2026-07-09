"""Diagnose why full embeddings can hurt combined failure detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.baselines import (
    _impute_nan,
    choose_cv_folds,
    compute_deployable_uncertainty_features,
    create_bad_case_label,
    evaluate_feature_set_binary,
    load_embeddings,
    neighbor_features_vs_reference,
)
from src.analysis.geometry import load_failure_table
from src.utils.io import ensure_dir

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
}


def _cv_binary_auroc(
    x: np.ndarray,
    labels: np.ndarray,
    classifier_name: str,
    random_state: int = 42,
) -> tuple[float, float, int]:
    """Out-of-fold AUROC and AUPRC for a pre-built feature matrix."""
    labels = np.asarray(labels, dtype=int)
    cv_folds = choose_cv_folds(labels)
    if cv_folds < 2:
        return float("nan"), float("nan"), 0

    proba = np.zeros(len(labels), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels):
        x_train = _impute_nan(x[train_idx])
        x_test = _impute_nan(x[test_idx])
        if classifier_name == "logistic_regression":
            model = LogisticRegression(max_iter=2000, random_state=random_state)
        elif classifier_name == "logistic_strong_reg":
            model = LogisticRegression(C=0.05, max_iter=2000, random_state=random_state)
        elif classifier_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200, random_state=random_state, n_jobs=1
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(x_train, labels[train_idx])
        proba[test_idx] = pipe.predict_proba(x_test)[:, 1]

    return (
        float(roc_auc_score(labels, proba)),
        float(average_precision_score(labels, proba)),
        cv_folds,
    )


def _select_top_dims_by_mi(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Pick embedding column indices using mutual information on the train fold only."""
    x_train = embeddings[train_idx]
    y_train = labels[train_idx]
    if len(np.unique(y_train)) < 2:
        return np.arange(min(top_k, embeddings.shape[1]))
    scores = mutual_info_classif(x_train, y_train, random_state=42)
    order = np.argsort(scores)[::-1]
    return order[: min(top_k, embeddings.shape[1])]


def _build_fold_features(
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    knn_neighbors: int,
    embedding_transform: str,
    top_k: int | None,
    include_neighbor: bool,
    pca_components: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build train/test feature matrices for one CV fold.

    embedding_transform:
      - full: raw 128-d embeddings
      - pca: PCA on train-fold embeddings
      - top_k: mutual-info top-k dims from train fold
      - none: no embedding block (uncertainty [+ neighbor] only)
    """
    emb_train = embeddings[train_idx]
    emb_test = embeddings[test_idx]
    n_emb_features = 0

    if embedding_transform == "none":
        emb_part_train = np.zeros((len(train_idx), 0), dtype=np.float32)
        emb_part_test = np.zeros((len(test_idx), 0), dtype=np.float32)
    elif embedding_transform == "full":
        emb_part_train = emb_train
        emb_part_test = emb_test
        n_emb_features = emb_train.shape[1]
    elif embedding_transform == "pca":
        n_comp = min(pca_components or 20, emb_train.shape[0], emb_train.shape[1])
        emb_scaler = StandardScaler().fit(emb_train)
        pca = PCA(n_components=n_comp, random_state=42)
        emb_part_train = pca.fit_transform(emb_scaler.transform(emb_train))
        emb_part_test = pca.transform(emb_scaler.transform(emb_test))
        n_emb_features = n_comp
    elif embedding_transform == "top_k":
        cols = _select_top_dims_by_mi(embeddings, labels, train_idx, top_k or 10)
        emb_part_train = emb_train[:, cols]
        emb_part_test = emb_test[:, cols]
        n_emb_features = len(cols)
    else:
        raise ValueError(f"Unknown embedding_transform: {embedding_transform}")

    if include_neighbor:
        neighbor_train = neighbor_features_vs_reference(
            emb_train, emb_train, n_neighbors=knn_neighbors
        )
        neighbor_test = neighbor_features_vs_reference(
            emb_test, emb_train, n_neighbors=knn_neighbors
        )
    else:
        neighbor_train = np.zeros((len(train_idx), 0), dtype=np.float32)
        neighbor_test = np.zeros((len(test_idx), 0), dtype=np.float32)

    x_train = np.concatenate(
        [uncertainty[train_idx], emb_part_train, neighbor_train], axis=1
    ).astype(np.float32)
    x_test = np.concatenate(
        [uncertainty[test_idx], emb_part_test, neighbor_test], axis=1
    ).astype(np.float32)
    n_features = x_train.shape[1]
    return x_train, x_test, n_features


def evaluate_embedding_variant(
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifier_name: str,
    embedding_transform: str,
    include_neighbor: bool = True,
    top_k: int | None = None,
    pca_components: int | None = None,
    knn_neighbors: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """CV evaluation for a custom uncertainty + embedding feature recipe."""
    labels = np.asarray(labels, dtype=int)
    cv_folds = choose_cv_folds(labels)
    if cv_folds < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "cv_folds": 0, "n_features": 0}

    proba = np.zeros(len(labels), dtype=np.float64)
    n_features = 0
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels):
        x_train, x_test, n_features = _build_fold_features(
            uncertainty,
            embeddings,
            labels,
            train_idx,
            test_idx,
            knn_neighbors,
            embedding_transform,
            top_k,
            include_neighbor,
            pca_components,
        )
        x_train = _impute_nan(x_train)
        x_test = _impute_nan(x_test)

        if classifier_name == "logistic_regression":
            model = LogisticRegression(max_iter=2000, random_state=random_state)
        elif classifier_name == "logistic_strong_reg":
            model = LogisticRegression(C=0.05, max_iter=2000, random_state=random_state)
        elif classifier_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200, random_state=random_state, n_jobs=1
            )
        else:
            raise ValueError(classifier_name)

        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(x_train, labels[train_idx])
        proba[test_idx] = pipe.predict_proba(x_test)[:, 1]

    return {
        "auroc": float(roc_auc_score(labels, proba)),
        "auprc": float(average_precision_score(labels, proba)),
        "cv_folds": cv_folds,
        "n_features": n_features,
    }


def embedding_dimension_sweep(
    uncertainty: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifier_name: str = "logistic_regression",
    top_k_values: list[int] | None = None,
    knn_neighbors: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """AUROC vs number of top mutual-information embedding dimensions."""
    if top_k_values is None:
        top_k_values = [1, 2, 3, 5, 10, 15, 20, 30, 50, 128]
    rows = []
    for k in top_k_values:
        result = evaluate_embedding_variant(
            uncertainty,
            embeddings,
            labels,
            classifier_name,
            embedding_transform="top_k",
            top_k=k,
            include_neighbor=True,
            knn_neighbors=knn_neighbors,
            random_state=random_state,
        )
        rows.append({"top_k_dims": k, **result})
    return pd.DataFrame(rows)


def analyze_embedding_hurting(
    failure_table_path: str | Path,
    output_dir: str | Path,
    dice_threshold: float = 0.80,
    knn_neighbors: int = 10,
    random_state: int = 42,
    run_label: str = "",
) -> pd.DataFrame:
    """
    Run diagnostic ablations to understand when/why embeddings hurt combined AUROC.
    """
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    failure_df = load_failure_table(failure_table_path)
    embeddings = load_embeddings(failure_df)
    uncertainty, _ = compute_deployable_uncertainty_features(failure_df)
    dice = failure_df["dice"].values.astype(np.float64)
    labels, effective_threshold = create_bad_case_label(dice, threshold=dice_threshold)

    rows: list[dict[str, Any]] = []

    def _record(name: str, classifier: str, result: dict[str, Any], **extra: Any) -> None:
        rows.append(
            {
                "run": run_label,
                "variant": name,
                "classifier": classifier,
                "n_cases": len(labels),
                "n_bad_cases": int(labels.sum()),
                "dice_threshold": effective_threshold,
                **result,
                **extra,
            }
        )

    # Replicate standard baselines.
    for feature_set in ("uncertainty_only", "geometry_only", "combined"):
        for clf in ("logistic_regression", "random_forest"):
            result = evaluate_feature_set_binary(
                feature_set,
                uncertainty,
                embeddings,
                labels,
                clf,
                knn_neighbors=knn_neighbors,
                random_state=random_state,
            )
            _record(f"baseline_{feature_set}", clf, result)

    # Dimensionality / representation ablations (logistic unless noted).
    ablations: list[tuple[str, str, dict[str, Any]]] = [
        ("unc_plus_full128_plus_knn", "logistic_regression", {"embedding_transform": "full", "include_neighbor": True}),
        ("unc_plus_pca10_plus_knn", "logistic_regression", {"embedding_transform": "pca", "pca_components": 10, "include_neighbor": True}),
        ("unc_plus_pca20_plus_knn", "logistic_regression", {"embedding_transform": "pca", "pca_components": 20, "include_neighbor": True}),
        ("unc_plus_top5_plus_knn", "logistic_regression", {"embedding_transform": "top_k", "top_k": 5, "include_neighbor": True}),
        ("unc_plus_top10_plus_knn", "logistic_regression", {"embedding_transform": "top_k", "top_k": 10, "include_neighbor": True}),
        ("unc_plus_top20_plus_knn", "logistic_regression", {"embedding_transform": "top_k", "top_k": 20, "include_neighbor": True}),
        ("unc_plus_full128_no_knn", "logistic_regression", {"embedding_transform": "full", "include_neighbor": False}),
        ("unc_plus_pca20_no_knn", "logistic_regression", {"embedding_transform": "pca", "pca_components": 20, "include_neighbor": False}),
        ("unc_plus_knn_only", "logistic_regression", {"embedding_transform": "none", "include_neighbor": True}),
        ("unc_plus_full128_plus_knn", "random_forest", {"embedding_transform": "full", "include_neighbor": True}),
        ("unc_plus_pca20_plus_knn", "random_forest", {"embedding_transform": "pca", "pca_components": 20, "include_neighbor": True}),
        ("unc_plus_top10_plus_knn", "random_forest", {"embedding_transform": "top_k", "top_k": 10, "include_neighbor": True}),
        ("unc_plus_full128_plus_knn", "logistic_strong_reg", {"embedding_transform": "full", "include_neighbor": True}),
        ("unc_plus_pca20_plus_knn", "logistic_strong_reg", {"embedding_transform": "pca", "pca_components": 20, "include_neighbor": True}),
    ]

    for variant, clf, kwargs in ablations:
        result = evaluate_embedding_variant(
            uncertainty,
            embeddings,
            labels,
            clf,
            knn_neighbors=knn_neighbors,
            random_state=random_state,
            **kwargs,
        )
        _record(variant, clf, result, **{k: v for k, v in kwargs.items() if k != "embedding_transform"})

    results_df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    results_df.to_csv(output_dir / "embedding_hurting_ablations.csv", index=False)

    # Top-k sweep.
    sweep_lr = embedding_dimension_sweep(
        uncertainty, embeddings, labels, "logistic_regression", random_state=random_state
    )
    sweep_rf = embedding_dimension_sweep(
        uncertainty, embeddings, labels, "random_forest", random_state=random_state
    )
    sweep_lr["classifier"] = "logistic_regression"
    sweep_rf["classifier"] = "random_forest"
    sweep_df = pd.concat([sweep_lr, sweep_rf], ignore_index=True)
    sweep_df.to_csv(output_dir / "embedding_topk_sweep.csv", index=False)

    # Permutation importance on full combined features (full-data fit, explanatory only).
    from src.analysis.baselines import assemble_feature_matrix

    neighbor_all = neighbor_features_vs_reference(
        embeddings, embeddings, n_neighbors=knn_neighbors
    )
    x_full = _impute_nan(
        assemble_feature_matrix("combined", uncertainty, embeddings, neighbor_all)
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]
    )
    pipe.fit(x_full, labels)
    perm = permutation_importance(
        pipe, x_full, labels, n_repeats=15, random_state=random_state, n_jobs=1
    )
    n_unc = uncertainty.shape[1]
    n_emb = embeddings.shape[1]
    feature_names = (
        [f"unc_{i}" for i in range(n_unc)]
        + [f"emb_{i}" for i in range(n_emb)]
        + ["knn_mean", "knn_std", "knn_min", "knn_sim"]
    )
    perm_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(output_dir / "combined_permutation_importance.csv", index=False)

    emb_perm = perm_df[perm_df["feature"].str.startswith("emb_")].copy()
    emb_perm["dimension"] = emb_perm["feature"].str.replace("emb_", "", regex=False).astype(int)
    n_informative = int((emb_perm["importance_mean"] > 0.001).sum())
    top10_frac = float(
        emb_perm.head(10)["importance_mean"].sum()
        / (emb_perm["importance_mean"].sum() + 1e-12)
    )

    # Figure: top-k sweep.
    plt.rcParams.update(PUBLICATION_RC)
    fig, ax = plt.subplots(figsize=(6, 4))
    for clf, style in [("logistic_regression", "o-"), ("random_forest", "s--")]:
        sub = sweep_df[sweep_df["classifier"] == clf]
        ax.plot(sub["top_k_dims"], sub["auroc"], style, label=clf.replace("_", " "))
    unc_auroc = results_df.loc[
        results_df["variant"] == "baseline_uncertainty_only", "auroc"
    ].max()
    comb_auroc = results_df.loc[
        results_df["variant"] == "baseline_combined", "auroc"
    ].max()
    ax.axhline(unc_auroc, color="C0", linestyle=":", alpha=0.7, label="uncertainty-only (best)")
    ax.axhline(comb_auroc, color="C2", linestyle=":", alpha=0.7, label="combined full (best)")
    ax.set_xlabel("Top-k embedding dimensions (mutual information, train-fold)")
    ax.set_ylabel("CV AUROC")
    ax.set_title(f"Does reducing embedding dimensionality help? (n={len(labels)} val)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "topk_dimension_sweep.png", bbox_inches="tight")
    plt.close(fig)

    # Figure: ablation bar chart (logistic variants only).
    logistic = results_df[results_df["classifier"] == "logistic_regression"].copy()
    logistic = logistic[~logistic["variant"].str.startswith("baseline_geometry")]
    plot_variants = logistic[
        logistic["variant"].isin(
            [
                "baseline_uncertainty_only",
                "baseline_combined",
                "unc_plus_pca10_plus_knn",
                "unc_plus_pca20_plus_knn",
                "unc_plus_top5_plus_knn",
                "unc_plus_top10_plus_knn",
                "unc_plus_top20_plus_knn",
                "unc_plus_knn_only",
                "unc_plus_full128_no_knn",
            ]
        )
    ].sort_values("auroc", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(plot_variants["variant"], plot_variants["auroc"], color="steelblue")
    ax.set_xlabel("CV AUROC")
    ax.set_title("Embedding ablations (logistic regression)")
    fig.tight_layout()
    fig.savefig(figures_dir / "ablation_auroc_logistic.png", bbox_inches="tight")
    plt.close(fig)

    # Markdown report.
    unc_baseline = float(
        results_df.loc[results_df["variant"] == "baseline_uncertainty_only", "auroc"].iloc[0]
    )
    comb_baseline = float(
        results_df.loc[results_df["variant"] == "baseline_combined", "auroc"].iloc[0]
    )
    best_row = results_df.iloc[0]
    pca20_row = results_df[
        (results_df["variant"] == "unc_plus_pca20_plus_knn")
        & (results_df["classifier"] == "logistic_regression")
    ]
    pca20_auroc = float(pca20_row["auroc"].iloc[0]) if len(pca20_row) else float("nan")
    top10_row = results_df[
        (results_df["variant"] == "unc_plus_top10_plus_knn")
        & (results_df["classifier"] == "logistic_regression")
    ]
    top10_auroc = float(top10_row["auroc"].iloc[0]) if len(top10_row) else float("nan")

    report = f"""# Why Are Embeddings Hurting? — Diagnostic Report

**Run:** {run_label or "unspecified"}
**Validation cases:** {len(labels)} | **Bad cases (Dice < {effective_threshold:.2f}):** {int(labels.sum())}

## Headline comparison

| Configuration | Classifier | AUROC | # features |
|---------------|------------|-------|------------|
| Uncertainty only | logistic | {unc_baseline:.3f} | 14 |
| Combined (full 128-d + kNN) | logistic | {comb_baseline:.3f} | 146 |
| Δ combined − uncertainty | | **{comb_baseline - unc_baseline:+.3f}** | |

## Key diagnostic questions

### 1. Are embedding features too high-dimensional?

Full combined uses **{int(results_df.loc[results_df['variant']=='unc_plus_full128_plus_knn', 'n_features'].iloc[0]) if len(results_df.loc[results_df['variant']=='unc_plus_full128_plus_knn']) else 146}** features
(14 uncertainty + 128 embedding + 4 kNN) vs **14** uncertainty-only.

Permutation importance on the full combined model: **{n_informative}** / 128 embedding
dimensions have mean importance > 0.001; top-10 dims account for **{top10_frac:.1%}** of
total embedding importance mass.

### 2. Does PCA to 10–20 dimensions help?

| Variant | AUROC |
|---------|-------|
| unc + PCA-10 + kNN | {float(results_df[(results_df.variant=='unc_plus_pca10_plus_knn')&(results_df.classifier=='logistic_regression')]['auroc'].iloc[0]) if len(results_df[(results_df.variant=='unc_plus_pca10_plus_knn')&(results_df.classifier=='logistic_regression')]) else float('nan'):.3f} |
| unc + PCA-20 + kNN | {pca20_auroc:.3f} |
| unc + full 128 + kNN | {comb_baseline:.3f} |

### 3. Does a nonlinear classifier use them better?

| Variant | Logistic AUROC | Random forest AUROC |
|---------|----------------|---------------------|
| Combined full | {comb_baseline:.3f} | {float(results_df[(results_df.variant=='baseline_combined')&(results_df.classifier=='random_forest')]['auroc'].iloc[0]):.3f} |
| PCA-20 + kNN | {pca20_auroc:.3f} | {float(results_df[(results_df.variant=='unc_plus_pca20_plus_knn')&(results_df.classifier=='random_forest')]['auroc'].iloc[0]) if len(results_df[(results_df.variant=='unc_plus_pca20_plus_knn')&(results_df.classifier=='random_forest')]) else float('nan'):.3f} |

### 4. Are only a handful of dimensions useful?

Top-10 mutual-information dims + kNN (logistic): **{top10_auroc:.3f}** AUROC.
See `embedding_topk_sweep.csv` and `figures/topk_dimension_sweep.png`.

### 5. Does adding all 128 dimensions overwhelm the classifier?

Stronger L2 logistic (C=0.05) on full combined:
**{float(results_df[(results_df.variant=='unc_plus_full128_plus_knn')&(results_df.classifier=='logistic_strong_reg')]['auroc'].iloc[0]) if len(results_df[(results_df.variant=='unc_plus_full128_plus_knn')&(results_df.classifier=='logistic_strong_reg')]) else float('nan'):.3f}** AUROC.

kNN neighbor features **without** raw embeddings (logistic):
**{float(results_df[(results_df.variant=='unc_plus_knn_only')&(results_df.classifier=='logistic_regression')]['auroc'].iloc[0]) if len(results_df[(results_df.variant=='unc_plus_knn_only')&(results_df.classifier=='logistic_regression')]) else float('nan'):.3f}** AUROC.

## Interpretation (diagnostic, not prescriptive)

See full ablation table in `embedding_hurting_ablations.csv`.

Best observed variant in this sweep: **{best_row['variant']}** ({best_row['classifier']}) AUROC={best_row['auroc']:.3f}.
"""
    (output_dir / "embedding_hurting_report.md").write_text(report)

    return results_df
