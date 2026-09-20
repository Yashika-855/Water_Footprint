#!/usr/bin/env python
"""Stage 2 - wheat crop calendar and the seven-day cultivation rule.

Output: data/interim/wheat_crop_calendar.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.config import get_logger, load_config
from src.crop_calendar import attach_calendar
from src.grids import from_output_columns
from src.io_utils import load_table

log = get_logger("02_prepare_calendar")

if __name__ == "__main__":
    cfg = load_config()
    ccfg = cfg["crop_calendar"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheat-type", default=ccfg["default_wheat_type"],
                    choices=ccfg["wheat_types"],
                    help="swh = spring wheat, wwh = winter wheat")
    ap.add_argument("--irrigation", default=ccfg["default_irrigation"],
                    choices=ccfg["irrigation"],
                    help="noirr = rainfed, firr = irrigated")
    args = ap.parse_args()

    wf = from_output_columns(
        load_table(Path(cfg["paths"]["interim"]) / cfg["output"]["wf_table"]))
    out = attach_calendar(wf, cfg, args.wheat_type, args.irrigation)
    log.info("Calendar attached: %d rows, mean season %.0f days, "
             "mean growing months %.2f",
             len(out), out["season_length_days"].mean(),
             out["n_growing_months"].mean())
