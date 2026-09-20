"""MODULE 9 - feature selection: Pearson screening + XGBoost importance -> top 20."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import get_logger
from .io_utils import save_table

log = get_logger(__name__)


def pearson_screen(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Pearson r between every predictor and the target, ranked by |r|."""
    r = X.apply(lambda col: col.corr(y))
    out = (pd.DataFrame({"feature": r.index, "pearson_r": r.values})
           .assign(abs_r=lambda d: d["pearson_r"].abs())
           .sort_values("abs_r", ascending=False)
           .reset_index(drop=True))
    log.info("Pearson screening done; strongest |r| = %.3f (%s)",
             out.loc[0, "abs_r"], out.loc[0, "feature"])
    return out


def xgb_importance(cfg: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Gain-based feature importance from an XGBoost regressor."""
    from xgboost import XGBRegressor

    params = dict(cfg["feature_selection"]["xgb_params"])
    model = XGBRegressor(random_state=cfg["project"]["random_state"],
                         n_jobs=-1, **params)
    model.fit(X, y)
    imp = (pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    log.info("XGBoost importance done; top feature = %s (%.4f)",
             imp.loc[0, "feature"], imp.loc[0, "importance"])
    return imp


def select_features(cfg: dict, X: pd.DataFrame, y: pd.Series,
                    target_col: str) -> tuple[list[str], pd.DataFrame]:
    """Combine both signals and return the paper's 20 most significant features."""
    n = cfg["feature_selection"]["n_features"]
    thr = cfg["feature_selection"].get("pearson_threshold", 0.0)

    pear = pearson_screen(X, y)
    kept = pear.loc[pear["abs_r"] >= thr, "feature"].tolist()
    if len(kept) < n:
        log.warning("Pearson threshold %.2f left only %d features; using all.", thr, len(kept))
        kept = list(X.columns)

    imp = xgb_importance(cfg, X[kept], y)
    ranking = imp.merge(pear[["feature", "pearson_r", "abs_r"]], on="feature", how="left")
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected"] = ranking["rank"] <= n

    top = ranking.loc[ranking["selected"], "feature"].tolist()
    log.info("Selected top %d features: %s", n, top)

    save_table(ranking, Path(cfg["paths"]["tables"]) / f"feature_ranking_{target_col}.csv")
    save_table(ranking[ranking["selected"]],
               Path(cfg["paths"]["tables"]) / f"top{n}_features_{target_col}.csv")
    return top, ranking
