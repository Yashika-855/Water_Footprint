#!/usr/bin/env python
"""Stage 6 - cleaning, 80:20 split, then Pearson + XGBoost -> 20 features.

Outputs:
    data/processed/clean_<target>.csv
    data/processed/split_<target>.joblib
    data/processed/selected_features.csv
    metadata/selected_features.txt
    results/metrics/feature_ranking_<target>.csv
    results/figures/correlation_<target>.png, xgb_importance_<target>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import joblib
import pandas as pd

from src.config import get_logger, load_config
from src.feature_selection import select_features
from src.io_utils import load_table, save_table
from src.plots import correlation_heatmap, importance_plot
from src.preprocess import clean, split_and_scale

log = get_logger("06_select_features")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all",
                    help="green_wf | blue_wf | total_wf | all")
    args = ap.parse_args()

    cfg = load_config()
    proc = Path(cfg["paths"]["processed"])
    full = load_table(proc / "features_full.csv")
    targets = (list(cfg["target"]["variables"].values())
               if args.target == "all" else [args.target])

    selected, rows = {}, []
    for target in targets:
        log.info("=== %s ===", target)
        clean_df = clean(cfg, full, target)
        save_table(clean_df, proc / f"clean_{target}.csv")

        split = split_and_scale(cfg, clean_df, target)
        joblib.dump(split, proc / f"split_{target}.joblib")
        log.info("split_%s.joblib saved (%d train / %d test)",
                 target, len(split["X_train"]), len(split["X_test"]))

        top, ranking = select_features(cfg, split["X_train"], split["y_train"], target)
        selected[target] = top
        rows += [{"target": target, "rank": i + 1, "feature": f}
                 for i, f in enumerate(top)]

        correlation_heatmap(split["X_train"], split["y_train"],
                            cfg["paths"]["figures"], target)
        importance_plot(ranking, cfg["paths"]["figures"], target,
                        cfg["feature_selection"]["n_features"])

    save_table(pd.DataFrame(rows), proc / cfg["output"]["selected_table"])
    with open(proc / "selected_features.json", "w") as fh:
        json.dump(selected, fh, indent=2)
    with open(Path(cfg["paths"]["metadata"]) / "selected_features.txt", "w") as fh:
        for target, feats in selected.items():
            fh.write(f"# {target}\n" + "\n".join(feats) + "\n\n")
    log.info("Selected features written to %s and metadata/selected_features.txt",
             cfg["output"]["selected_table"])
