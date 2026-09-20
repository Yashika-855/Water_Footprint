import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.evaluation import compute_metrics


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y, y)
    assert m["MAE"] == 0 and m["R2"] == 1.0 and m["n"] == 3


def test_metrics_are_finite():
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 2000, 200)
    p = y + rng.normal(0, 50, 200)
    m = compute_metrics(y, p)
    assert all(np.isfinite(v) for k, v in m.items() if k != "n")
    assert m["RMSE"] >= m["MAE"]


def test_mape_is_a_percentage_by_default():
    y = np.array([100.0, 200.0])
    p = np.array([110.0, 180.0])          # 10% error on both
    m = compute_metrics(y, p, mape_as_percent=True)
    assert 9.9 < m["MAPE"] < 10.1
    m_frac = compute_metrics(y, p, mape_as_percent=False)
    assert 0.099 < m_frac["MAPE"] < 0.101


def test_zero_targets_excluded_from_mape():
    y = np.array([0.0, 100.0])
    p = np.array([5.0, 110.0])
    m = compute_metrics(y, p)
    assert m["n_excluded_from_MAPE"] == 1
