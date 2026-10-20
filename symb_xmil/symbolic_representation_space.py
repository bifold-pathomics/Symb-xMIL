"""Symbolic representation space.

Each bag X is represented by phi(X), its alignment scores r_X(q, f) over all
queries, read from `best_queries.csv`. phi(X) is projected to 2D with PCA,
colored by q*(X), and optionally clustered with KMeans.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, silhouette_score


METADATA_COLUMNS = ["best_query", "best_alignment_score", "cluster"]


def load_symbolic_results(best_queries: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Return the best-query table in the same structure as ``run_symbolic_xmil``'s ``best_df``.

    Index ``slide_id``; columns ``best_query``, ``best_alignment_score`` and one
    alignment score per query. Accepts ``best_df`` itself or a path to `best_queries.csv`.
    """
    if isinstance(best_queries, pd.DataFrame):
        return best_queries
    return pd.read_csv(best_queries, index_col="slide_id")


def get_query_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """phi(X): the alignment scores of all queries, one row per bag (index ``slide_id``)."""
    matrix = df.drop(columns=METADATA_COLUMNS, errors="ignore")
    if matrix.shape[1] == 0:
        raise ValueError("No alignment-score columns found.")
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def add_kmeans_clusters(
    df: pd.DataFrame,
    query_matrix: pd.DataFrame,
    k_range: Iterable[int] = (2, 3, 4, 5, 6),
    label_col: str = "best_query",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add KMeans cluster labels; k maximizes silhouette + homogeneity w.r.t. q*(X)."""
    n_samples = len(query_matrix)
    valid_k = [int(k) for k in k_range if 2 <= int(k) < n_samples]
    if not valid_k:
        raise ValueError("KMeans needs at least 3 samples and a valid k_range.")

    rows = []
    labels_ref = df[label_col].astype(str) if label_col in df.columns else None

    for k in valid_k:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(query_matrix)

        sil = silhouette_score(query_matrix, labels)
        homo = homogeneity_score(labels_ref, labels) if labels_ref is not None else 0.0
        rows.append(
            {
                "k": k,
                "silhouette": sil,
                "homogeneity": homo,
                "combined_score": sil + homo,
            }
        )

    report = pd.DataFrame(rows)
    best_k = int(report.loc[report["combined_score"].idxmax(), "k"])
    final_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)

    enriched = df.copy()
    enriched["cluster"] = final_kmeans.fit_predict(query_matrix)
    return enriched, report


def project_symbolic_representation_space(
    query_matrix: pd.DataFrame,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the symbolic representations phi(X) to two PCA dimensions."""
    n_samples, n_features = query_matrix.shape
    if n_samples < 2 or n_features < 1:
        raise ValueError("PCA needs at least 2 bags and 1 query.")

    n_components = min(2, n_samples, n_features)
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(query_matrix)

    if n_components == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(n_samples)])
        explained = np.array([pca.explained_variance_ratio_[0], 0.0])
    else:
        explained = pca.explained_variance_ratio_

    return coords, explained


def plot_symbolic_representation_space(
    best_queries: str | Path | pd.DataFrame,
    min_query_count: int | None = None,
    use_kmeans: bool = False,
    output_path: str | Path | None = None,
    title: str = "Symbolic Representation Space",
    figsize: tuple[float, float] = (10, 8),
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """PCA plot of the symbolic representation space.

    Args:
        best_queries:      ``best_df`` from ``run_symbolic_xmil`` or a path to `best_queries.csv`.
        min_query_count:   q*(X) with fewer bags are shown as "Others". None: show all.
        use_kmeans:        Add KMeans cluster labels.
        output_path:       Where to save the plot.

    Returns:
        (plot_df, enriched_df, cluster_report); cluster_report is None without KMeans.
    """
    df = load_symbolic_results(best_queries)
    query_matrix = get_query_matrix(df)

    cluster_report = None
    enriched_df = df
    if use_kmeans:
        enriched_df, cluster_report = add_kmeans_clusters(
            df,
            query_matrix,
            random_state=random_state,
        )

    coords, explained = project_symbolic_representation_space(query_matrix, random_state=random_state)
    plot_df = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "best_query": df["best_query"].astype(str),
        },
        index=df.index,
    )
    if use_kmeans:
        plot_df["cluster"] = enriched_df["cluster"].values

    _draw_symbolic_representation_space(
        plot_df,
        explained,
        min_query_count=min_query_count,
        output_path=output_path,
        title=title,
        figsize=figsize,
    )

    return plot_df, enriched_df, cluster_report


def _draw_symbolic_representation_space(
    plot_df: pd.DataFrame,
    explained: np.ndarray,
    min_query_count: int | None,
    output_path: str | Path | None,
    title: str,
    figsize: tuple[float, float],
) -> None:
    """Draw and save the symbolic representation space scatter plot."""
    query_counts = plot_df["best_query"].value_counts()
    if min_query_count is None:
        visible_queries = query_counts.index.tolist()
    else:
        visible_queries = query_counts[query_counts >= min_query_count].index.tolist()

    cmap = cm.get_cmap("tab20", max(len(visible_queries), 1))
    color_map = {
        query: mcolors.to_hex(cmap(i))
        for i, query in enumerate(visible_queries)
    }

    fig, ax = plt.subplots(figsize=figsize)
    others_plotted = False

    for query, group in plot_df.groupby("best_query"):
        if query not in visible_queries:
            ax.scatter(
                group["PC1"],
                group["PC2"],
                color="lightgray",
                label="Others" if not others_plotted else None,
                alpha=0.65,
                edgecolors="none",
                zorder=1,
            )
            others_plotted = True

    for query, group in plot_df.groupby("best_query"):
        if query in visible_queries:
            ax.scatter(
                group["PC1"],
                group["PC2"],
                color=color_map[query],
                label=f"{query} (n={len(group)})",
                alpha=0.9,
                edgecolors="black",
                linewidths=0.3,
                zorder=3,
            )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var.)")
    ax.legend(title="Best Query", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if output_path is None:
        output_path = "pca_symbolic_representation_space.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "add_kmeans_clusters",
    "get_query_matrix",
    "load_symbolic_results",
    "plot_symbolic_representation_space",
    "project_symbolic_representation_space",
]
