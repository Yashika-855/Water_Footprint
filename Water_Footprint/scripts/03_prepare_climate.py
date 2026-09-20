#!/usr/bin/env python
"""Stage 3 - daily GSWP3-W5E5 -> monthly climatology at the wheat locations.

Run `--variable pr` first to validate the pipeline on ONE variable before
downloading the remaining six (README section 18).

Output: data/interim/climate_features_2010_2019.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.climate import build_climate_features, monthly_climatology
from src.config import get_logger, load_config
from src.grids import from_output_columns
from src.io_utils import load_table

log = get_logger("03_prepare_climate")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variable", help="process a single variable as a dry run")
    args = ap.parse_args()

    cfg = load_config()
    points = from_output_columns(
        load_table(Path(cfg["paths"]["interim"]) / cfg["output"]["locations_table"]))

    if args.variable:
        clim = monthly_climatology(cfg, args.variable)
        log.info("Dry run OK for %s -> dims %s, unit %s, monthly op %s",
                 args.variable, dict(clim.sizes), clim.attrs.get("units"),
                 clim.attrs.get("monthly_aggregation"))
        log.info("Verify these against the NetCDF attributes before trusting "
                 "the unit conversion in config.yaml.")
    else:
        build_climate_features(cfg, points)
