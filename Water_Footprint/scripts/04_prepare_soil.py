#!/usr/bin/env python
"""Stage 4 - soil attributes per grid cell from the FAO DSMW.

If config soil.attributes is empty this script deliberately FAILS with the
available column list printed, so you choose attributes from the paper rather
than inventing them.

Output: data/interim/soil_features.csv
"""
from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

from src.config import get_logger, load_config
from src.grids import from_output_columns
from src.io_utils import load_table
from src.soil import build_soil_features

log = get_logger("04_prepare_soil")

if __name__ == "__main__":
    cfg = load_config()
    points = from_output_columns(
        load_table(Path(cfg["paths"]["interim"]) / cfg["output"]["locations_table"]))
    out = build_soil_features(cfg, points)
    log.info("Soil features: %d columns", out.shape[1])
