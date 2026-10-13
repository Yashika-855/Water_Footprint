"""Module 13 - Model evaluation for the wheat water-footprint ML pipeline.

Metrics (as defined in the project guide):
    MAE, MSE, MAPE (%), Test Score (= 100 - MAPE %), R2

Prediction files are expected to be named
    results/predictions/{target}_wf_predictions_{method}.csv
e.g. green_wf_predictions_kmeans.csv, total_wf_predictions_hierarchical.csv

Column names are detected automatically (actual/y_true, predicted/y_pred, ...).
Only y_true and y_pred are mandatory; cluster, latitude, longitude and split
are used when present.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: safe in scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

FILENAME_PATTERN = re.compile(r"^(?P<target>[a-z]+)_wf_predictions_(?P<method>.+)\.csv$", re.I)
EXPECTED_COMBINATIONS = [
    (target, method) for target in ("green", "blue", "total") for method in ("kmeans", "hierarchical")
]
# The guide expects roughly 3,510 test rows (20 % of ~17,552). Much more => train rows mixed in.
EXPECTED_TEST_ROWS = 3510

# Reference values reported by the base paper. Comparison only - never present as your results.
PAPER_REFERENCE = pd.DataFrame(
    {
        "target": ["green", "blue", "total"],
        "paper_MAE": [278.57, 590.21, 208.6],
        "paper_MAPE_pct": [4.04, 17.12, 2.7],
        "paper_TestScore_pct": [95.96, 82.88, 97.35],
        "paper_R2": [0.52, 0.40, 0.69],
    }
)


# --------------------------------------------------------------------------- #
# Column detection
# --------------------------------------------------------------------------- #
def _find_column(columns: list[str], exact: tuple[str, ...], contains: tuple[str, ...] = (),
                 exclude: tuple[str, ...] = ()) -> str | None:
    """Return the single matching column, None if absent, raise if ambiguous."""
    lowered = {c: c.strip().lower() for c in columns}
    hits = [c for c, low in lowered.items() if low in exact]
    if not hits:
        hits = [
            c for c, low in lowered.items()
            if any(token in low for token in contains) and not any(bad in low for bad in exclude)
        ]
    if len(hits) > 1:
        raise ValueError(f"Ambiguous columns {hits}; rename the right one to {exact[0]!r}.")
    return hits[0] if hits else None


def standardize_columns(df: pd.DataFrame, source: str = "file") -> pd.DataFrame:
    """Rename whatever the file uses to: y_true, y_pred, cluster, latitude, longitude, split."""
    cols = list(df.columns)
    mapping = {
        "y_true": _find_column(cols, ("y_true", "actual", "observed", "true"), ("actual", "observed", "y_true", "true")),
        "y_pred": _find_column(cols, ("y_pred", "predicted", "prediction", "pred"), ("pred",), exclude=("cluster",)),
        "cluster": _find_column(cols, ("cluster", "cluster_id", "cluster_label", "label")),
        "latitude": _find_column(cols, ("latitude", "lat")),
        "longitude": _find_column(cols, ("longitude", "lon", "lng")),
        "split": _find_column(cols, ("split", "set", "subset")),
    }
    missing = [name for name in ("y_true", "y_pred") if mapping[name] is None]
    if missing:
        raise ValueError(
            f"{source}: could not find {missing}. Columns present: {cols}. "
            "Rename the actual/predicted columns or tell your mentor which they are."
        )
    rename = {old: new for new, old in mapping.items() if old is not None and old != new}
    return df.rename(columns=rename)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mape_pct(y_true, y_pred, min_abs_true: float = 1e-6) -> tuple[float, int]:
    """MAPE in percent. Rows with |y_true| <= min_abs_true are excluded
    (division by ~0 would explode the score). Returns (mape, n_excluded)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > min_abs_true
    n_excluded = int((~mask).sum())
    if not mask.any():
        return float("nan"), n_excluded
    value = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    return float(value), n_excluded


def compute_metrics(y_true, y_pred, min_abs_true: float = 1e-6) -> dict:
    """Return the project metrics plus two robust companions.

    MAPE / Test Score follow the guide. Rows with |y_true| <= min_abs_true are
    left out of MAPE only (default 1e-6 = only exact zeros). WMAPE and MedianAPE
    are reported next to it because MAPE explodes when actual values are tiny.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length.")

    mape, n_excluded = mape_pct(y_true, y_pred, min_abs_true)
    mask = np.abs(y_true) > min_abs_true
    denominator = np.abs(y_true).sum()
    wmape = float(np.abs(y_true - y_pred).sum() / denominator * 100.0) if denominator > 0 else float("nan")
    median_ape = (
        float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
        if mask.any() else float("nan")
    )
    return {
        "n_samples": int(y_true.size),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAPE_pct": mape,
        "TestScore_pct": 100.0 - mape,
        "R2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
        "WMAPE_pct": wmape,
        "MedianAPE_pct": median_ape,
        "n_excluded_from_MAPE": n_excluded,
    }


# --------------------------------------------------------------------------- #
# Loading and aggregation
# --------------------------------------------------------------------------- #
def parse_filename(path: Path) -> tuple[str, str]:
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(
            f"Unexpected file name '{path.name}'. Expected {{target}}_wf_predictions_{{method}}.csv"
        )
    return match["target"].lower(), match["method"].lower()


def load_predictions(path: str | Path, split: str = "test") -> pd.DataFrame:
    """Load one predictions file, standardise columns, keep the requested split."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    target, method = parse_filename(path)
    df = standardize_columns(pd.read_csv(path, encoding="utf-8-sig"), source=path.name)

    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split]
    elif len(df) > 1.5 * EXPECTED_TEST_ROWS:
        logger.warning(
            "%s has %d rows and no split column - the guide expects ~%d test rows. "
            "Train rows may be mixed in; metrics would be too optimistic.",
            path.name, len(df), EXPECTED_TEST_ROWS,
        )

    before = len(df)
    df = df.dropna(subset=["y_true", "y_pred"])
    if len(df) < before:
        logger.warning("%s: dropped %d rows with NaN y_true/y_pred", path.name, before - len(df))
    if df.empty:
        raise ValueError(f"{path.name}: no rows left to evaluate for split='{split}'.")

    df = df.copy()
    df["target"] = target
    df["method"] = method
    return df


def evaluate_method(df: pd.DataFrame, min_abs_true: float = 1e-6) -> pd.DataFrame:
    """Overall metrics, plus per-cluster metrics when a cluster column exists."""
    rows = [{"scope": "overall", "cluster": "all", **compute_metrics(df["y_true"], df["y_pred"], min_abs_true)}]
    if "cluster" in df.columns:
        for cluster_id, part in df.groupby("cluster"):
            rows.append(
                {"scope": "cluster", "cluster": cluster_id, **compute_metrics(part["y_true"], part["y_pred"], min_abs_true)}
            )
    return pd.DataFrame(rows)


def compare_with_paper(final_results: pd.DataFrame) -> pd.DataFrame:
    """Join student overall results with the paper's reference values."""
    merged = final_results.merge(PAPER_REFERENCE, on="target", how="left")
    merged["delta_MAE"] = merged["MAE"] - merged["paper_MAE"]
    merged["delta_MAPE_pct"] = merged["MAPE_pct"] - merged["paper_MAPE_pct"]
    merged["delta_R2"] = merged["R2"] - merged["paper_R2"]
    return merged


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_actual_vs_predicted(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["y_true"], df["y_pred"], s=6, alpha=0.4)
    lo = min(df["y_true"].min(), df["y_pred"].min())
    hi = max(df["y_true"].max(), df["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="Perfect prediction")
    ax.set_xlabel("Actual WF (m³/t)")
    ax.set_ylabel("Predicted WF (m³/t)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residuals(df: pd.DataFrame, title: str, out_path: Path) -> None:
    residual = df["y_pred"] - df["y_true"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(df["y_pred"], residual, s=6, alpha=0.4)
    axes[0].axhline(0, color="r", ls="--", lw=1)
    axes[0].set_xlabel("Predicted WF (m³/t)")
    axes[0].set_ylabel("Residual (pred - actual)")
    axes[0].set_title("Residuals vs predicted")
    axes[1].hist(residual, bins=50)
    axes[1].set_xlabel("Residual (m³/t)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spatial_error(df: pd.DataFrame, title: str, out_path: Path) -> bool:
    """Needs latitude/longitude columns; returns False (and plots nothing) if absent."""
    if not {"latitude", "longitude"} <= set(df.columns):
        return False
    error = df["y_pred"] - df["y_true"]
    limit = float(np.nanpercentile(np.abs(error), 95)) or 1.0  # clip outliers for readability
    fig, ax = plt.subplots(figsize=(11, 5))
    sc = ax.scatter(df["longitude"], df["latitude"], c=error, s=8, cmap="coolwarm", vmin=-limit, vmax=limit)
    fig.colorbar(sc, ax=ax, label="Prediction error (m³/t)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True