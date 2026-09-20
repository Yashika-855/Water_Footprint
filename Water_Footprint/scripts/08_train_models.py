#!/usr/bin/env python
"""Stage 8 - cluster-wise AdaBoost regression, one model per cluster.

Outputs:
    models/adaboost_<method>_<target>.joblib
    results/predictions/<target>_predictions_<method>.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import joblib
import pandas as pd

from src.config import get_logger, load_config
from src.models import ClusterwiseAdaBoost

log = get_logger("08_train_models")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="total_wf",
                    help="green_wf | blue_wf | total_wf | all")
    args = ap.parse_args()

    cfg = load_config()
    proc = Path(cfg["paths"]["processed"])
    targets = (list(cfg["target"]["variables"].values())
               if args.target == "all" else [args.target])

    log.info("Hyperparameter source: %s", cfg["model"]["hyperparameter_source"])

    for target in targets:
        split = joblib.load(proc / f"split_{target}.joblib")
        with open(proc / "selected_features.json") as fh:
            top = json.load(fh)[target]

        label_path = proc / f"cluster_labels_{target}.joblib"
        if not label_path.exists():
            log.warning("%s missing - run 07_cluster.py --target %s first",
                        label_path.name, target)
            continue
        labels = joblib.load(label_path)

        X_tr, X_te = split["X_train"][top], split["X_test"][top]
        y_tr, y_te = split["y_train"], split["y_test"]

        for method in cfg["clustering"]["methods"]:
            model = ClusterwiseAdaBoost(cfg, method, target)
            model.fit(X_tr, y_tr, labels[method]["train"])
            y_pred = model.predict(X_te, labels[method]["test"])
            model.save(cfg["paths"]["models"])

            pd.DataFrame({"y_true": y_te.to_numpy(), "y_pred": y_pred,
                          "cluster": labels[method]["test"]}).to_csv(
                Path(cfg["paths"]["predictions"]) /
                f"{target}_predictions_{method}.csv", index=False)
            log.info("[%s/%s] predictions written (%d rows)",
                     method, target, len(y_pred))
