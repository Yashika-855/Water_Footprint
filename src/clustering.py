"""MODULE 10 - elbow / silhouette diagnostics and the two K=5 clusterings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from .config import get_logger
from .io_utils import save_table

log = get_logger(__name__)


def elbow_and_silhouette(cfg: dict, X: pd.DataFrame,
                         sample_for_silhouette: int = 5000) -> pd.DataFrame:
    """Inertia and silhouette score across the configured K range."""
    rs = cfg["project"]["random_state"]
    rows = []
    rng = np.random.default_rng(rs)
    sample_idx = (rng.choice(len(X), sample_for_silhouette, replace=False)
                  if len(X) > sample_for_silhouette else np.arange(len(X)))

    for k in cfg["clustering"]["k_range"]:
        km = KMeans(n_clusters=k, random_state=rs, n_init=10).fit(X)
        sil = silhouette_score(X.iloc[sample_idx], km.labels_[sample_idx])
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        log.info("k=%-2d  inertia=%12.1f  silhouette=%.4f", k, km.inertia_, sil)

    out = pd.DataFrame(rows)
    save_table(out, Path(cfg["paths"]["tables"]) / "cluster_diagnostics.csv")
    return out


def fit_clusters(cfg: dict, X: pd.DataFrame, method: str = "kmeans") -> np.ndarray:
    """Fit K-means or Ward hierarchical clustering at the paper's K."""
    k = cfg["clustering"]["k_final"]
    rs = cfg["project"]["random_state"]

    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=rs, n_init=10)
    elif method == "hierarchical":
        model = AgglomerativeClustering(
            n_clusters=k, linkage=cfg["clustering"]["hierarchical_linkage"])
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = model.fit_predict(X)
    sizes = pd.Series(labels).value_counts().sort_index()
    log.info("%s (K=%d) cluster sizes: %s", method, k, sizes.to_dict())
    return labels


def cluster_size_table(cfg: dict, labels_by_method: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for method, labels in labels_by_method.items():
        vc = pd.Series(labels).value_counts().sort_index()
        for cluster, n in vc.items():
            rows.append({"method": method, "cluster": int(cluster), "n": int(n),
                         "share": round(n / len(labels), 4)})
    out = pd.DataFrame(rows)
    save_table(out, Path(cfg["paths"]["tables"]) / "cluster_sizes.csv")
    return out
