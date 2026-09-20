"""Figures: elbow, silhouette, correlation heatmap, importance, parity, maps."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import get_logger

log = get_logger(__name__)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9})


def _save(fig, out_dir, name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path); plt.close(fig)
    log.info("Figure -> %s", path.name)
    return path


def elbow_plot(diag: pd.DataFrame, out_dir, k_final: int | None = None):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(diag["k"], diag["inertia"], "o-")
    if k_final:
        ax.axvline(k_final, ls="--", c="crimson", lw=1, label=f"K = {k_final}")
        ax.legend()
    ax.set_xlabel("Number of clusters (K)"); ax.set_ylabel("Inertia")
    ax.set_title("Elbow method")
    return _save(fig, out_dir, "elbow.png")


def silhouette_plot(diag: pd.DataFrame, out_dir, k_final: int | None = None):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(diag["k"], diag["silhouette"], "o-", color="seagreen")
    if k_final:
        ax.axvline(k_final, ls="--", c="crimson", lw=1, label=f"K = {k_final}")
        ax.legend()
    ax.set_xlabel("Number of clusters (K)"); ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette analysis")
    return _save(fig, out_dir, "silhouette.png")


def correlation_heatmap(X: pd.DataFrame, y: pd.Series, out_dir,
                        target_col: str, top_n: int = 25):
    r = X.apply(lambda c: c.corr(y)).sort_values(key=np.abs, ascending=False)[:top_n]
    fig, ax = plt.subplots(figsize=(4.5, 0.26 * len(r) + 1))
    colors = ["#c0392b" if v < 0 else "#2874a6" for v in r.values]
    ax.barh(range(len(r)), r.values, color=colors)
    ax.set_yticks(range(len(r))); ax.set_yticklabels(r.index)
    ax.invert_yaxis(); ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel(f"Pearson r with {target_col}")
    ax.set_title(f"Top {top_n} correlations")
    return _save(fig, out_dir, f"correlation_{target_col}.png")


def importance_plot(ranking: pd.DataFrame, out_dir, target_col: str, top_n: int = 20):
    d = ranking.head(top_n)
    fig, ax = plt.subplots(figsize=(4.5, 0.26 * len(d) + 1))
    ax.barh(range(len(d)), d["importance"], color="#8e44ad")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["feature"])
    ax.invert_yaxis(); ax.set_xlabel("XGBoost gain importance")
    ax.set_title(f"Top {top_n} features - {target_col}")
    return _save(fig, out_dir, f"xgb_importance_{target_col}.png")


def parity_plot(y_true, y_pred, out_dir, target_col: str, method: str, r2: float):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3, edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"Observed {target_col}"); ax.set_ylabel(f"Predicted {target_col}")
    ax.set_title(f"{method}  (R2 = {r2:.3f})")
    return _save(fig, out_dir, f"actual_vs_predicted_{target_col}_{method}.png")


def cluster_map(df: pd.DataFrame, labels, out_dir, method: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(df["lon"], df["lat"], c=labels, s=2, cmap="tab10")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Cluster membership - {method}")
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="cluster")
    return _save(fig, out_dir, f"cluster_map_{method}.png")
