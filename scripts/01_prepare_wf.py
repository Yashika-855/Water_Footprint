#!/usr/bin/env python
"""Stage 1 - build the wheat water-footprint target table.

Output: data/interim/wf_wheat_2010_2019.csv
        data/interim/wheat_locations.csv   (drives climate subsetting)
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from src.config import get_logger, load_config
from src.wf import build_target

log = get_logger("01_prepare_wf")

if __name__ == "__main__":
    cfg = load_config()
    df = build_target(cfg)
    cols = list(cfg["target"]["variables"].values())
    log.info("Target table ready: %d wheat locations", len(df))
    log.info("\n%s", df[cols].describe())
