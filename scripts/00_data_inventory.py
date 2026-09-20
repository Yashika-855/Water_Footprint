#!/usr/bin/env python
"""Inventory every raw file before any modelling starts.

Writes results/metrics/data_inventory.csv with file name, size, format,
dimensions, coordinate ranges, longitude convention and variables.
"""
from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from src.config import get_logger, load_config
from src.io_utils import save_table

log = get_logger("inventory")


def inspect_netcdf(path: Path) -> dict:
    import xarray as xr
    with xr.open_dataset(path, decode_times=False) as ds:
        lon_name = next((c for c in ("lon", "longitude", "x") if c in ds.coords), None)
        lat_name = next((c for c in ("lat", "latitude", "y") if c in ds.coords), None)
        lon_conv = ""
        if lon_name is not None:
            lon_conv = "0..360" if float(ds[lon_name].max()) > 180 else "-180..180"
        return {
            "dims": str(dict(ds.sizes)),
            "variables": ", ".join(list(ds.data_vars)),
            "coords": ", ".join(list(ds.coords)),
            "lon_convention": lon_conv,
            "lat_range": (f"{float(ds[lat_name].min()):.3f}.."
                          f"{float(ds[lat_name].max()):.3f}" if lat_name else ""),
            "units": str({v: ds[v].attrs.get("units", "n/a")
                          for v in list(ds.data_vars)[:4]}),
            "missing_value": str({
                v: ds[v].attrs.get("_FillValue", ds[v].attrs.get("missing_value", "n/a"))
                for v in list(ds.data_vars)[:4]}),
        }


def inspect_vector(path: Path) -> dict:
    import geopandas as gpd
    gdf = gpd.read_file(path, rows=5)
    return {"dims": f"{len(gdf.columns)} columns (5-row peek)",
            "variables": ", ".join(gdf.columns[:25]), "coords": str(gdf.crs),
            "lon_convention": "", "lat_range": "", "units": "",
            "missing_value": "n/a"}


if __name__ == "__main__":
    cfg = load_config()
    sources = {"wf": cfg["paths"]["raw_wf"], "climate": cfg["paths"]["raw_climate"],
               "crop_calendar": cfg["paths"]["raw_calendar"],
               "soil": cfg["paths"]["raw_soil"]}

    rows = []
    for source, folder in sources.items():
        for p in sorted(Path(folder).rglob("*")):
            if p.is_dir() or p.name == ".gitkeep":
                continue
            row = {"source": source, "file": p.name,
                   "relative_path": str(p.relative_to(cfg["_root"])),
                   "size_mb": round(p.stat().st_size / 1e6, 2),
                   "format": p.suffix.lstrip(".")}
            try:
                if p.suffix in (".nc", ".nc4"):
                    row |= inspect_netcdf(p)
                elif p.suffix == ".shp":
                    row |= inspect_vector(p)
                elif p.suffix in (".csv", ".tsv"):
                    head = pd.read_csv(p, nrows=5)
                    row |= {"dims": f"{head.shape[1]} columns",
                            "variables": ", ".join(head.columns[:25])}
            except Exception as exc:                        # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
                log.warning("Could not inspect %s: %s", p.name, exc)
            rows.append(row)
            log.info("%-14s %-48s %8.2f MB", source, p.name[:48], row["size_mb"])

    if not rows:
        log.error("No raw files found. Run 00_download_data.py and follow "
                  "docs/data_collection.md first.")
    else:
        inv = pd.DataFrame(rows)
        save_table(inv, Path(cfg["paths"]["tables"]) / "data_inventory.csv")
        log.info("Inventory complete: %d files, %.1f MB total",
                 len(inv), inv["size_mb"].sum())
