"""MODULE 11 - cluster-wise AdaBoost regression.

One AdaBoostRegressor per cluster per clustering method per target. The weak
learner is a DecisionTreeRegressor, as described in the paper.

Honesty note carried from the guide: the paper does not publish a complete
hyperparameter table. config.yaml ships sklearn defaults and records that fact
in `model.hyperparameter_source`. Do not relabel invented values as "paper
settings".
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from .config import get_logger

log = get_logger(__name__)


def make_model(cfg: dict) -> AdaBoostRegressor:
    p = cfg["model"]["params"]
    base = DecisionTreeRegressor(max_depth=p["base_max_depth"],
                                 random_state=cfg["project"]["random_state"])
    try:                                     # sklearn >= 1.2
        return AdaBoostRegressor(estimator=base, n_estimators=p["n_estimators"],
                                 learning_rate=p["learning_rate"], loss=p["loss"],
                                 random_state=cfg["project"]["random_state"])
    except TypeError:                        # older sklearn
        return AdaBoostRegressor(base_estimator=base, n_estimators=p["n_estimators"],
                                 learning_rate=p["learning_rate"], loss=p["loss"],
                                 random_state=cfg["project"]["random_state"])


class ClusterwiseAdaBoost:
    """Fit one AdaBoost model per cluster; predict by routing rows to their model.

    A global fallback model is also fitted so that a test row landing in a
    cluster with too few training samples still receives a prediction (and the
    fallback usage is counted, not hidden).
    """

    def __init__(self, cfg: dict, method: str, target: str, min_cluster_size: int = 20):
        self.cfg, self.method, self.target = cfg, method, target
        self.min_cluster_size = min_cluster_size
        self.models: dict[int, AdaBoostRegressor] = {}
        self.fallback: AdaBoostRegressor | None = None
        self.fallback_rows = 0

    def fit(self, X: pd.DataFrame, y: pd.Series, labels: np.ndarray) -> "ClusterwiseAdaBoost":
        self.fallback = make_model(self.cfg).fit(X, y)
        for c in np.unique(labels):
            mask = labels == c
            if mask.sum() < self.min_cluster_size:
                log.warning("[%s/%s] cluster %d has only %d rows - using fallback",
                            self.method, self.target, c, int(mask.sum()))
                continue
            self.models[int(c)] = make_model(self.cfg).fit(X[mask], y[mask])
            log.info("[%s/%s] cluster %d trained on %d rows",
                     self.method, self.target, c, int(mask.sum()))
        return self

    def predict(self, X: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        out = np.empty(len(X), dtype=float)
        self.fallback_rows = 0
        for c in np.unique(labels):
            mask = labels == c
            model = self.models.get(int(c))
            if model is None:
                model = self.fallback
                self.fallback_rows += int(mask.sum())
            out[mask] = model.predict(X[mask])
        if self.fallback_rows:
            log.info("[%s/%s] %d test rows used the fallback model",
                     self.method, self.target, self.fallback_rows)
        return out

    def save(self, models_dir: str | Path) -> Path:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"adaboost_{self.method}_{self.target}.joblib"
        joblib.dump({"models": self.models, "fallback": self.fallback,
                     "method": self.method, "target": self.target,
                     "params": self.cfg["model"]["params"],
                     "hyperparameter_source": self.cfg["model"]["hyperparameter_source"]},
                    path)
        log.info("Saved %s", path.name)
        return path
