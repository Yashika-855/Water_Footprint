#!/usr/bin/env python
"""Stage 9 - MAE, MSE, RMSE, MAPE, R2; per-cluster breakdown; parity plots.

Outputs:
    results/metrics/<method>_<target>_metrics.csv
    results/metrics/model_comparison.csv
    results/figures/actual_vs_predicted_<target>_<method>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from src.config import get_logger, load_config
from src.evaluation import ResultsCollector, compute_metrics, per_cluster_metrics
from src.io_utils import save_table
from src.plots import parity_plot

log = get_logger("09_evaluate")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all")
    args = ap.parse_args()

    cfg = load_config()
    pred_dir = Path(cfg["paths"]["predictions"])
    metrics_dir = Path(cfg["paths"]["tables"])
    as_pct = cfg["evaluation"]["mape_as_percent"]
    targets = (list(cfg["target"]["variables"].values())
               if args.target == "all" else [args.target])

    collector = ResultsCollector()
    for target in targets:
        for method in cfg["clustering"]["methods"]:
            path = pred_dir / f"{target}_predictions_{method}.csv"
            if not path.exists():
                log.warning("missing %s - run 08_train_models.py first", path.name)
                continue
            df = pd.read_csv(path)
            metrics = compute_metrics(df["y_true"], df["y_pred"], as_pct)
            collector.add(target, method, metrics,
                          note=cfg["model"]["hyperparameter_source"])

            save_table(pd.DataFrame([metrics]),
                       metrics_dir / f"{method}_{target}_metrics.csv")
            save_table(per_cluster_metrics(df["y_true"], df["y_pred"],
                                           df["cluster"].to_numpy()),
                       metrics_dir / f"{method}_{target}_per_cluster.csv")
            parity_plot(df["y_true"], df["y_pred"], cfg["paths"]["figures"],
                        target, method, metrics["R2"])

    if collector.rows:
        collector.save(metrics_dir)
        log.info("\n%s", collector.to_frame().to_string(index=False))
        log.info("RQ2 check: compare clustered R2 against a single global model "
                 "before claiming clustering helps.")
    else:
        log.error("No predictions found. Run stages 06-08 first.")
