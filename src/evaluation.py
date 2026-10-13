"""MODULE 12 - metrics and comparison tables.

Metrics required by the guide: MAE, MSE, MAPE, test score, R2.
"test score" is reported as sklearn's .score() equivalent, i.e. R2 on the test
set; RMSE is added because it is in the paper's units and is easy to interpret.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

from .config import get_logger
from .io_utils import save_table

log = get_logger(__name__)


def compute_metrics(y_true, y_pred, mape_as_percent: bool = True) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    nonzero = y_true != 0
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero])
                      * (100.0 if mape_as_percent else 1.0)),
        "n_excluded_from_MAPE": int((~nonzero).sum()),
        "R2": float(r2_score(y_true, y_pred)),
        "test_score": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def per_cluster_metrics(y_true, y_pred, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < 2:
            continue
        rows.append({"cluster": int(c), **compute_metrics(np.asarray(y_true)[m],
                                                          np.asarray(y_pred)[m])})
    return pd.DataFrame(rows)


class ResultsCollector:
    """Accumulate every (target, method) result into one comparison table."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, target: str, method: str, metrics: dict, note: str = "") -> None:
        self.rows.append({"target": target, "clustering": method, **metrics, "note": note})
        log.info("[%s / %s] R2=%.4f  MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%",
                 target, method, metrics["R2"], metrics["MAE"],
                 metrics["RMSE"], metrics["MAPE"])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, tables_dir: str | Path) -> Path:
        df = self.to_frame()
        return save_table(df, Path(tables_dir) / "model_comparison.csv")
