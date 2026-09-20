#!/usr/bin/env python
"""Stage 5 - spatial integration + the candidate feature table.

Output: data/processed/master_wheat_dataset.csv
        data/processed/features_full.csv
        metadata/feature_dictionary.csv
"""
from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

from src.config import get_logger, load_config
from src.features import build_feature_table
from src.grids import from_output_columns
from src.integration import integrate
from src.io_utils import load_table

log = get_logger("05_build_master")

if __name__ == "__main__":
    cfg = load_config()
    interim = Path(cfg["paths"]["interim"])
    out = cfg["output"]

    wf = from_output_columns(load_table(interim / out["wf_table"]))
    cal = from_output_columns(load_table(interim / out["calendar_table"]))
    clim = from_output_columns(load_table(interim / out["climate_table"]))

    soil_path = interim / out["soil_table"]
    soil = load_table(soil_path) if soil_path.exists() else None
    if soil is None:
        log.warning("%s missing - continuing without soil. Record this as a "
                    "deviation in docs/methodology.md.", out["soil_table"])

    master = integrate(cfg, wf, cal, clim, soil)
    log.info("Master dataset: %d rows x %d cols", *master.shape)

    table, dictionary = build_feature_table(cfg, master)
    n_pred = len([c for c in table.columns
                  if c not in cfg["target"]["variables"].values()])
    log.info("Built %d candidate features. Paper reports 102.", n_pred)
    log.info("Do NOT claim exact reproduction unless you verified the list "
             "against the paper's supplementary material.")
    log.info("\n%s", dictionary.groupby("group").size())
