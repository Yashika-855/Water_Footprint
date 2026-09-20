#!/usr/bin/env python
"""Stage 7 - elbow + silhouette diagnostics, then K-means and hierarchical (K=5).

Outputs:
    data/processed/cluster_assignments_kmeans.csv
    data/processed/cluster_assignments_hierarchical.csv
    data/processed/cluster_labels_<target>.joblib
    results/metrics/cluster_diagnostics.csv, cluster_sizes.csv
    results/figures/elbow.png, silhouette.png, cluster_map.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd

from src.clustering import cluster_size_table, elbow_and_silhouette, fit_clusters
from src.config import get_logger, load_config
from src.io_utils import load_table, save_table
from src.plots import cluster_map, elbow_plot, silhouette_plot

log = get_logger("07_cluster")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="total_wf",
                    help="which split's feature space to cluster on")
    ap.add_argument("--skip-diagnostics", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    proc = Path(cfg["paths"]["processed"])

    split = joblib.load(proc / f"split_{args.target}.joblib")
    with open(proc / "selected_features.json") as fh:
        top = json.load(fh)[args.target]

    X_tr, X_te = split["X_train"][top], split["X_test"][top]
    clean_df = load_table(proc / f"clean_{args.target}.csv")

    if not args.skip_diagnostics:
        diag = elbow_and_silhouette(cfg, X_tr)
        elbow_plot(diag, cfg["paths"]["figures"], cfg["clustering"]["k_final"])
        silhouette_plot(diag, cfg["paths"]["figures"], cfg["clustering"]["k_final"])
        log.info("Confirm K=%d is defensible from these plots before continuing.",
                 cfg["clustering"]["k_final"])

    labels = {}
    for method in cfg["clustering"]["methods"]:
        tr_labels = fit_clusters(cfg, X_tr, method)

        if method == "kmeans":
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=cfg["clustering"]["k_final"],
                        random_state=cfg["project"]["random_state"],
                        n_init=10).fit(X_tr)
            te_labels = km.predict(X_te)
        else:
            # AgglomerativeClustering has no predict(); assign test rows to the
            # nearest training-cluster centroid. DOCUMENT this in the report.
            cents = np.vstack([X_tr.values[tr_labels == c].mean(axis=0)
                               for c in np.unique(tr_labels)])
            d = ((X_te.values[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
            te_labels = d.argmin(axis=1)
            log.info("Hierarchical test labels assigned by nearest centroid "
                     "(deviation - record it).")

        labels[method] = {"train": tr_labels, "test": te_labels}

        # latitude | longitude | cluster, as specified in the README
        coords = clean_df.loc[split["idx_train"], ["lat", "lon"]] \
            if "lat" in clean_df.columns else None
        if coords is not None:
            assign = coords.rename(columns={"lat": "latitude", "lon": "longitude"})
            assign["cluster"] = tr_labels
            assign["split"] = "train"
            te_coords = clean_df.loc[split["idx_test"], ["lat", "lon"]].rename(
                columns={"lat": "latitude", "lon": "longitude"})
            te_coords["cluster"] = te_labels
            te_coords["split"] = "test"
            save_table(pd.concat([assign, te_coords], ignore_index=True),
                       proc / f"cluster_assignments_{method}.csv")
            cluster_map(coords, tr_labels, cfg["paths"]["figures"], method)

    joblib.dump(labels, proc / f"cluster_labels_{args.target}.joblib")
    cluster_size_table(cfg, {m: v["train"] for m, v in labels.items()})
    log.info("Cluster labels saved for %s", args.target)
