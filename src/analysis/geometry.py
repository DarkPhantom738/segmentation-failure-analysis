"""Latent-space geometry analysis for segmentation failures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from umap import UMAP

from src.analysis.failure_taxonomy import assign_dominant_failure_label
from src.utils.io import ensure_dir


def load_failure_table(failure_table_path: str | Path) -> pd.DataFrame:
    """Load Milestone 3 failure metrics and attach dominant failure labels."""
    failure_df = pd.read_csv(failure_table_path)
    failure_df["dominant_failure_label"] = failure_df.apply(
        lambda row: assign_dominant_failure_label(row.to_dict()),
        axis=1,
    )
    return failure_df


def load_embeddings(failure_df: pd.DataFrame) -> np.ndarray:
    """Load bottleneck embeddings listed in the failure table."""
    embeddings = [np.load(path).astype(np.float32) for path in failure_df["path_embedding"]]
    return np.stack(embeddings, axis=0)


def preprocess_embeddings(
    embeddings: np.ndarray,
    pca_components: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize embeddings and optionally reduce dimensionality with PCA.

    PCA is applied only when the embedding dimension exceeds `pca_components`
    and there are at least two cases.
    """
    scaler = StandardScaler()
    standardized = scaler.fit_transform(embeddings)

    # PCA requires at least two cases to estimate meaningful components.
    if standardized.shape[0] >= 2 and standardized.shape[1] > pca_components:
        n_components = min(pca_components, standardized.shape[0], standardized.shape[1])
        reduced = PCA(n_components=n_components, random_state=random_state).fit_transform(
            standardized
        )
        return standardized, reduced

    return standardized, standardized


def run_umap(
    features: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project features to 2D with UMAP for visualization.

    For very small validation sets, falls back to the first two feature dimensions.
    """
    n_cases = features.shape[0]
    if n_cases < 4:
        # UMAP is unstable with only a few points; use PCA for visualization.
        if n_cases < 2:
            coords = np.zeros((n_cases, 2), dtype=np.float32)
            if n_cases == 1 and features.shape[1] >= 1:
                coords[0, 0] = float(features[0, 0])
            return coords

        from sklearn.decomposition import PCA

        n_components = min(2, features.shape[1], n_cases)
        return PCA(n_components=n_components, random_state=random_state).fit_transform(features)

    n_neighbors = min(15, max(2, n_cases - 1))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def build_knn_graph(
    features: np.ndarray,
    n_neighbors: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a k-nearest-neighbor graph in embedding space.

    Returns neighbor indices and distances for each case.
    """
    n_cases = features.shape[0]
    if n_cases <= 1:
        return np.zeros((n_cases, 0), dtype=int), np.zeros((n_cases, 0), dtype=float)

    k = min(n_neighbors, n_cases - 1)
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    model.fit(features)
    distances, indices = model.kneighbors(features)
    # Drop the self-neighbor at position 0.
    return indices[:, 1:], distances[:, 1:]


def knn_label_agreement(
    labels: np.ndarray,
    neighbor_indices: np.ndarray,
) -> float:
    """
    Mean fraction of k nearest neighbors sharing the same dominant failure label.

    This tests whether similar latent embeddings co-occur with similar failure types.
    """
    if neighbor_indices.size == 0:
        return float("nan")

    agreements: list[float] = []
    for case_index, neighbors in enumerate(neighbor_indices):
        neighbor_labels = labels[neighbors]
        agreements.append(float(np.mean(neighbor_labels == labels[case_index])))

    return float(np.mean(agreements))


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster cases in latent space with KMeans."""
    n_cases = features.shape[0]
    if n_cases == 0:
        return np.array([], dtype=int)

    n_clusters = min(max(n_clusters, 1), n_cases)
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        algorithm="lloyd",
    )
    return model.fit_predict(features)


def cluster_purity(labels: np.ndarray, cluster_ids: np.ndarray) -> float:
    """
    Weighted cluster purity relative to dominant failure labels.

    For each cluster, purity is the fraction of cases with the majority label.
    The overall score is weighted by cluster size.
    """
    if len(labels) == 0:
        return float("nan")

    total = len(labels)
    weighted_purity = 0.0
    for cluster_id in np.unique(cluster_ids):
        cluster_mask = cluster_ids == cluster_id
        cluster_labels = labels[cluster_mask]
        majority_count = int(pd.Series(cluster_labels).value_counts().max())
        weighted_purity += majority_count / total
    return float(weighted_purity)


def per_cluster_enrichment(
    failure_df: pd.DataFrame,
    cluster_ids: np.ndarray,
) -> pd.DataFrame:
    """Summarize each KMeans cluster with failure-label enrichment."""
    summary_df = failure_df.copy()
    summary_df["cluster_id"] = cluster_ids

    rows: list[dict[str, float | int | str]] = []
    for cluster_id in sorted(summary_df["cluster_id"].unique()):
        cluster_cases = summary_df[summary_df["cluster_id"] == cluster_id]
        label_counts = cluster_cases["dominant_failure_label"].value_counts()
        most_common_label = str(label_counts.idxmax())
        purity = float(label_counts.max() / len(cluster_cases))

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_cases": int(len(cluster_cases)),
                "most_common_failure_label": most_common_label,
                "purity": purity,
                "mean_dice": float(cluster_cases["dice"].mean()),
                "mean_entropy_error": float(cluster_cases["mean_entropy_error"].mean()),
                "mean_entropy_error_auroc": float(cluster_cases["entropy_error_auroc"].mean()),
            }
        )

    return pd.DataFrame(rows)


def build_umap_coordinates_table(
    failure_df: pd.DataFrame,
    umap_coords: np.ndarray,
) -> pd.DataFrame:
    """Create per-case UMAP coordinate table for downstream visualization."""
    return pd.DataFrame(
        {
            "case_id": failure_df["case_id"],
            "umap_x": umap_coords[:, 0],
            "umap_y": umap_coords[:, 1],
            "dominant_failure_label": failure_df["dominant_failure_label"],
            "dice": failure_df["dice"],
            "mean_entropy_error": failure_df["mean_entropy_error"],
            "entropy_error_auroc": failure_df["entropy_error_auroc"],
            "path_prediction": failure_df["path_prediction"],
            "path_ground_truth": failure_df["path_ground_truth"],
            "path_entropy": failure_df["path_entropy"],
        }
    )


def plot_umap_failure_labels(
    coordinates_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a static UMAP scatter plot colored by dominant failure label."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, axis = plt.subplots(figsize=(8, 6))
    labels = coordinates_df["dominant_failure_label"].unique()

    for label in sorted(labels):
        subset = coordinates_df[coordinates_df["dominant_failure_label"] == label]
        axis.scatter(
            subset["umap_x"],
            subset["umap_y"],
            label=label,
            alpha=0.85,
            s=60,
        )

    axis.set_title("UMAP of Bottleneck Embeddings by Dominant Failure Label")
    axis.set_xlabel("UMAP-1")
    axis.set_ylabel("UMAP-2")
    axis.legend(title="Failure label", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_geometry(
    failure_table_path: str | Path,
    output_dir: str | Path,
    knn_neighbors: int = 10,
    pca_components: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run Milestone 4 geometry analysis on validation-case embeddings.

    Returns:
        geometry_metrics, umap_coordinates, cluster_summary DataFrames
    """
    output_dir = ensure_dir(output_dir)
    failure_df = load_failure_table(failure_table_path)

    embeddings = load_embeddings(failure_df)
    standardized, model_features = preprocess_embeddings(
        embeddings,
        pca_components=pca_components,
        random_state=random_state,
    )

    umap_coords = run_umap(model_features, random_state=random_state)
    neighbor_indices, _ = build_knn_graph(standardized, n_neighbors=knn_neighbors)

    labels = failure_df["dominant_failure_label"].to_numpy()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    n_unique_labels = len(label_encoder.classes_)
    n_clusters = max(min(n_unique_labels, len(failure_df)), 1)
    cluster_ids = run_kmeans(model_features, n_clusters=n_clusters, random_state=random_state)

    # Silhouette measures cohesion of dominant failure labels in latent space.
    if n_unique_labels > 1 and len(failure_df) > n_unique_labels:
        silhouette = float(silhouette_score(model_features, encoded_labels))
    else:
        silhouette = float("nan")

    if n_unique_labels > 1 and len(np.unique(cluster_ids)) > 1:
        ari = float(adjusted_rand_score(labels, cluster_ids))
        nmi = float(normalized_mutual_info_score(labels, cluster_ids))
    else:
        ari = float("nan")
        nmi = float("nan")

    geometry_metrics = pd.DataFrame(
        [
            {
                "knn_label_agreement": knn_label_agreement(labels, neighbor_indices),
                "silhouette_score": silhouette,
                "adjusted_rand_index": ari,
                "normalized_mutual_information": nmi,
                "cluster_purity": cluster_purity(labels, cluster_ids),
            }
        ]
    )

    umap_coordinates = build_umap_coordinates_table(failure_df, umap_coords)
    cluster_summary = per_cluster_enrichment(failure_df, cluster_ids)

    geometry_metrics.to_csv(output_dir / "geometry_metrics.csv", index=False)
    umap_coordinates.to_csv(output_dir / "umap_coordinates.csv", index=False)
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)
    plot_umap_failure_labels(umap_coordinates, output_dir / "umap_failure_labels.png")

    return geometry_metrics, umap_coordinates, cluster_summary
