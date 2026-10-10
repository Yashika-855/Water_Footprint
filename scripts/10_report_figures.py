#!/usr/bin/env python
"""Module 14 - the required figures and tables not produced by stages 06-09.

Adds, from files already saved by earlier stages (nothing is retrained):

    #2  Missing-value report          results/metrics/missing_value_report.csv
    #7  Cluster-size chart            results/figures/cluster_sizes.png
    #9  Residual / error plots        results/figures/residuals_<target>_<method>.png
    #10 Model feature importance      results/figures/model_importance_<target>_<method>.png
    #11 Spatial prediction map        results/figures/spatial_prediction_<target>.png
    #12 Spatial error map             results/figures/spatial_error_<target>.png

Maps and importance plots use the better method per target (highest R2 in
model_comparison.csv). Residual plots are made for every method.

Run:  python scripts/10_report_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import get_logger, load_config
from src.io_utils import load_table, save_table

log = get_logger("10_report_figures")

INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
BLUE, ORANGE = "#2a6fbb", "#d9822b"          # K-means, hierarchical
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": "#bbbbbb"})
METHOD_COLOR = {"kmeans": BLUE, "hierarchical": ORANGE}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name)
    plt.close(fig)
    log.info("Figure -> %s", name)


# --------------------------------------------------------------------------- #
# #2 missing-value report
# --------------------------------------------------------------------------- #
def missing_value_report(cfg: dict) -> None:
    path = Path(cfg["paths"]["processed"]) / "features_full.csv"
    if not path.exists():
        log.warning("%s missing - skipping missing-value report", path.name)
        return
    df = load_table(path)
    rep = (pd.DataFrame({"column": df.columns,
                         "n_missing": df.isna().sum().values,
                         "pct_missing": (df.isna().mean() * 100).round(2).values})
           .sort_values("n_missing", ascending=False).reset_index(drop=True))
    rep.insert(0, "n_rows", len(df))
    save_table(rep, Path(cfg["paths"]["tables"]) / "missing_value_report.csv")
    n_bad = int((rep["n_missing"] > 0).sum())
    log.info("Missing values: %d of %d columns have gaps (rows=%d)",
             n_bad, len(rep), len(df))


# --------------------------------------------------------------------------- #
# #7 cluster-size chart
# --------------------------------------------------------------------------- #
def cluster_size_chart(cfg: dict) -> None:
    path = Path(cfg["paths"]["tables"]) / "cluster_sizes.csv"
    if not path.exists():
        log.warning("%s missing - run 07_cluster.py first", path.name)
        return
    cs = pd.read_csv(path)
    pivot = cs.pivot(index="cluster", columns="method", values="n").fillna(0)
    clusters = pivot.index.to_numpy()
    methods = [m for m in ("kmeans", "hierarchical") if m in pivot.columns]
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, m in enumerate(methods):
        x = np.arange(len(clusters)) + (i - (len(methods) - 1) / 2) * (width + 0.03)
        bars = ax.bar(x, pivot[m].to_numpy(), width, color=METHOD_COLOR[m], label=m)
        ax.bar_label(bars, fmt="%d", fontsize=7, color=MUTED, padding=2)
    ax.set_xticks(np.arange(len(clusters)))
    ax.set_xticklabels([f"Cluster {int(c)}" for c in clusters])
    ax.set_ylabel("Training locations")
    ax.set_title("Cluster sizes (K = 5)")
    ax.legend(frameon=False)
    _style(ax)
    ax.grid(axis="x", visible=False)
    _save(fig, Path(cfg["paths"]["figures"]), "cluster_sizes.png")


# --------------------------------------------------------------------------- #
# helpers shared by the per-target plots
# --------------------------------------------------------------------------- #
def best_method(cfg: dict, target: str) -> str | None:
    path = Path(cfg["paths"]["tables"]) / "model_comparison.csv"
    if not path.exists():
        return None
    mc = pd.read_csv(path)
    mc = mc[mc["target"] == target]
    return None if mc.empty else str(mc.sort_values("R2").iloc[-1]["clustering"])


def read_predictions(cfg: dict, target: str, method: str) -> pd.DataFrame | None:
    path = Path(cfg["paths"]["predictions"]) / f"{target}_predictions_{method}.csv"
    return pd.read_csv(path) if path.exists() else None


def test_coordinates(cfg: dict, target: str, n_expected: int) -> pd.DataFrame | None:
    """lat/lon of the test rows, in the same order as the prediction files.

    split_and_scale() saved `idx_test` as positions into clean_<target>.csv, and
    train_test_split keeps rows in index order, so the order matches.
    """
    proc = Path(cfg["paths"]["processed"])
    clean_p, split_p = proc / f"clean_{target}.csv", proc / f"split_{target}.joblib"
    if not (clean_p.exists() and split_p.exists()):
        log.warning("clean/split files for %s missing - skipping maps", target)
        return None
    clean = pd.read_csv(clean_p)
    split = joblib.load(split_p)
    cols = ("lat", "lon") if "lat" in clean.columns else ("latitude", "longitude")
    coords = clean.iloc[split["idx_test"]][list(cols)].copy()
    coords.columns = ["lat", "lon"]
    if len(coords) != n_expected:
        log.warning("%s: %d coordinates vs %d predictions - skipping maps",
                    target, len(coords), n_expected)
        return None
    return coords.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# #9 residual plot
# --------------------------------------------------------------------------- #
def residual_plot(cfg: dict, target: str, method: str, pred: pd.DataFrame) -> None:
    resid = pred["y_pred"] - pred["y_true"]          # + = over-prediction
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    a1.scatter(pred["y_pred"], resid, s=7, alpha=0.4, color=METHOD_COLOR[method],
               edgecolors="none")
    a1.axhline(0, color=INK, linewidth=1)
    a1.set_xlabel(f"Predicted {target}")
    a1.set_ylabel("Residual (predicted - observed)")
    a1.set_title("Residuals vs predicted")
    a2.hist(resid, bins=40, color=METHOD_COLOR[method])
    a2.axvline(0, color=INK, linewidth=1)
    a2.set_xlabel("Residual (m3/t)")
    a2.set_ylabel("Locations")
    a2.set_title(f"Error distribution (mean {resid.mean():.1f}, sd {resid.std():.1f})")
    for ax in (a1, a2):
        _style(ax)
    fig.suptitle(f"{target} - {method}", y=1.02)
    _save(fig, Path(cfg["paths"]["figures"]), f"residuals_{target}_{method}.png")


# --------------------------------------------------------------------------- #
# #11 / #12 spatial maps
# --------------------------------------------------------------------------- #
def _base_map(ax, coords: pd.DataFrame) -> None:
    ax.set_xlim(coords["lon"].min() - 5, coords["lon"].max() + 5)
    ax.set_ylim(coords["lat"].min() - 5, coords["lat"].max() + 5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    _style(ax)


def spatial_maps(cfg: dict, target: str, method: str, pred: pd.DataFrame,
                 coords: pd.DataFrame) -> None:
    obs, est = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    vmax = float(np.nanpercentile(np.concatenate([obs, est]), 98))   # one scale, outliers clipped
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True, sharey=True)
    for ax, vals, title in ((axes[0], obs, "Observed (ACEA)"),
                            (axes[1], est, f"Predicted ({method})")):
        sc = ax.scatter(coords["lon"], coords["lat"], c=vals, s=9, cmap="Blues",
                        vmin=0, vmax=vmax, edgecolors="none")
        ax.set_title(title)
        _base_map(ax, coords)
    fig.colorbar(sc, ax=axes, shrink=0.8, label=f"{target} (m3/t, clipped at 98th pct)")
    fig.suptitle(f"Spatial pattern of {target} - test locations", y=1.02)
    _save(fig, Path(cfg["paths"]["figures"]), f"spatial_prediction_{target}.png")

    resid = est - obs
    lim = float(np.nanpercentile(np.abs(resid), 95))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    sc = ax.scatter(coords["lon"], coords["lat"], c=resid, s=9, cmap="RdBu_r",
                    vmin=-lim, vmax=lim, edgecolors="none")
    _base_map(ax, coords)
    ax.set_title(f"Prediction error - {target} ({method})")
    fig.colorbar(sc, ax=ax, shrink=0.8,
                 label="Predicted - observed (m3/t)\nred = over, blue = under")
    _save(fig, Path(cfg["paths"]["figures"]), f"spatial_error_{target}.png")


# --------------------------------------------------------------------------- #
# #10 model feature importance
# --------------------------------------------------------------------------- #
def model_importance(cfg: dict, target: str, method: str, top_n: int = 20) -> None:
    mpath = Path(cfg["paths"]["models"]) / f"adaboost_{method}_{target}.joblib"
    fpath = Path(cfg["paths"]["processed"]) / "selected_features.json"
    if not (mpath.exists() and fpath.exists()):
        log.warning("model or selected_features.json missing for %s - skipping", target)
        return
    bundle = joblib.load(mpath)
    feats = json.loads(fpath.read_text())[target]
    imps = [m.feature_importances_ for m in bundle["models"].values()]
    if not imps:
        imps = [bundle["fallback"].feature_importances_]
    mean_imp = np.mean(imps, axis=0)
    if len(mean_imp) != len(feats):
        log.warning("importance length %d != %d features - skipping %s",
                    len(mean_imp), len(feats), target)
        return
    d = (pd.DataFrame({"feature": feats, "importance": mean_imp})
         .sort_values("importance", ascending=False).reset_index(drop=True))
    save_table(d, Path(cfg["paths"]["tables"]) / f"model_importance_{target}_{method}.csv")
    d = d.head(top_n)
    fig, ax = plt.subplots(figsize=(5, 0.26 * len(d) + 1))
    ax.barh(range(len(d)), d["importance"], color=METHOD_COLOR[method])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean AdaBoost importance across cluster models")
    ax.set_title(f"Model feature importance - {target} ({method})")
    _style(ax)
    ax.grid(axis="y", visible=False)
    _save(fig, Path(cfg["paths"]["figures"]), f"model_importance_{target}_{method}.png")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = load_config()
    missing_value_report(cfg)
    cluster_size_chart(cfg)

    for target in cfg["target"]["variables"].values():
        log.info("=== %s ===", target)
        for method in cfg["clustering"]["methods"]:
            pred = read_predictions(cfg, target, method)
            if pred is None:
                log.warning("no predictions for %s/%s - run 08_train_models.py", target, method)
                continue
            residual_plot(cfg, target, method, pred)

        best = best_method(cfg, target)
        if best is None:
            continue
        log.info("Best method for %s by R2: %s", target, best)
        pred = read_predictions(cfg, target, best)
        if pred is not None:
            coords = test_coordinates(cfg, target, len(pred))
            if coords is not None:
                spatial_maps(cfg, target, best, pred, coords)
        model_importance(cfg, target, best)

    log.info("Done. Figures in %s, tables in %s",
             cfg["paths"]["figures"], cfg["paths"]["tables"])
