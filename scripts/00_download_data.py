#!/usr/bin/env python
"""Download what can be automated; print instructions for the rest.

Auto:   GGCMI Phase 3 crop calendar (Zenodo 5062513, ~3 MB) via the Zenodo API.
Manual: water footprint (4TU), climate (ISIMIP, account required), soil (FAO).

Usage:  python scripts/00_download_data.py [--skip-calendar] [--all-crops]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import requests
from tqdm import tqdm

from src.config import get_logger, load_config

log = get_logger("download")

ZENODO_RECORD = "5062513"
WHEAT_KEYS = ("swh", "wwh", "wheat")


def download_crop_calendar(dest: Path, wheat_only: bool = True) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    api = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    log.info("Querying %s", api)
    meta = requests.get(api, timeout=60).json()

    files = meta.get("files", [])
    if wheat_only:
        files = [f for f in files
                 if any(k in f["key"].lower() for k in WHEAT_KEYS)] or files

    log.info("Downloading %d file(s) to %s", len(files), dest)
    for f in files:
        out = dest / f["key"]
        if out.exists():
            log.info("skip (exists): %s", out.name)
            continue
        r = requests.get(f["links"]["self"], stream=True, timeout=300)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True,
                                         desc=out.name[:40]) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk); bar.update(len(chunk))
    log.info("Crop calendar done.")


MANUAL = """
================================================================================
MANUAL DOWNLOADS   (full walk-through: docs/data_collection.md)
================================================================================

[1] WATER FOOTPRINT  ->  data/raw/wf/
    https://data.4tu.nl/datasets/7b45bcc6-686b-404d-a910-13c87156716a
    Get: readme.pdf                                   (~0.6 MB, READ FIRST)
         unit_wf_selected_crops_average_2010_2019.zip (~148 MB)
    Unzip, keep only the wheat NetCDF files, leave them unmodified.
    Record the dataset VERSION (v1/v2/v3) in metadata/DATA_SOURCES.csv.

[2] CLIMATE  ->  data/raw/climate/
    https://data.isimip.org   (free account required)
    ISIMIP3a/InputData/climate/atmosphere/obsclim/global/daily/historical/
    GSWP3-W5E5
    Variables: pr, sfcWind, tasmin, tasmax, hurs, rsds, ps
    For 2010-2019 you need the 2001_2010 AND 2011_2020 decade chunks of each.
    >>> Use the repository's CUTOUT service to request a bounding box.
    >>> Download `pr` first, validate, then fetch the remaining six.

[3] SOIL  ->  data/raw/soil/
    https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/
    faounesco-soil-map-of-the-world/en/
    Follow "Digital Soil Map of the World (Geonetwork)". Keep the .shp with its
    .dbf/.shx/.prj siblings and the attribute key.
    Then fill soil.attributes in config/config.yaml - do not invent variables.
================================================================================
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-calendar", action="store_true")
    ap.add_argument("--all-crops", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if not args.skip_calendar:
        download_crop_calendar(cfg["paths"]["raw_calendar"],
                               wheat_only=not args.all_crops)
    print(MANUAL)
