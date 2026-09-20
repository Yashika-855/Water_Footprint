"""MODULE 3 - GSWP3-W5E5 daily climate -> monthly features at wheat locations.

Source: ISIMIP Repository, ISIMIP3a / InputData / climate / atmosphere /
        obsclim / global / daily / historical / GSWP3-W5E5
        Daily, 0.5 deg, variables pr sfcWind tasmin tasmax hurs rsds ps.

Strategy that keeps this tractable on a student machine:
  1. Build the wheat location list FIRST (Module 2).
  2. Open each daily file lazily with dask.
  3. Subset to the bounding box of those locations.
  4. Aggregate to monthly with the per-variable operation from config.yaml.
  5. Average the monthly values over the study years -> 12 values per variable.
  6. Extract at the location points only.

Never load a global daily array into memory.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import get_logger
from .grids import nearest_cell_index, normalise_longitude, standardise_coord_names
from .io_utils import save_table

log = get_logger(__name__)

UNIT_CONVERSIONS = {
    "K_to_C": lambda da: da - 273.15,
    "kgm2s_to_mm": lambda da: da * 86400.0,   # kg m-2 s-1 -> mm/day
    None: lambda da: da,
}


def find_variable_files(raw_dir: str | Path, variable: str,
                        year_start: int, year_end: int) -> list[Path]:
    """Find GSWP3-W5E5 files for one variable overlapping the study years.

    ISIMIP names files like
      gswp3-w5e5_obsclim_pr_global_daily_2011_2020.nc
    so the trailing year pair is parsed to decide which decade chunks matter.
    """
    raw_dir = Path(raw_dir)
    hits: list[Path] = []
    for p in sorted(raw_dir.rglob("*.nc*")):
        name = p.name.lower()
        if f"_{variable.lower()}_" not in name:
            continue
        years = re.findall(r"(\d{4})_(\d{4})", name)
        if not years:
            hits.append(p)
            continue
        f_start, f_end = int(years[-1][0]), int(years[-1][1])
        if f_end >= year_start and f_start <= year_end:
            hits.append(p)
    if not hits:
        raise FileNotFoundError(
            f"No files for variable '{variable}' covering {year_start}-{year_end} "
            f"in {raw_dir}. See docs/DOWNLOAD_GUIDE.md section 3."
        )
    log.info("%s: %d file(s) -> %s", variable, len(hits), [h.name for h in hits])
    return hits


def monthly_climatology(cfg: dict, variable: str,
                        bbox: tuple[float, float, float, float] | None = None
                        ) -> xr.DataArray:
    """Daily -> monthly (per config op) -> multi-year monthly climatology.

    Returns a DataArray with dims (month, lat, lon), month = 1..12.
    """
    ccfg = cfg["climate"]
    spec = ccfg["variables"][variable]
    y0, y1 = cfg["target"]["year_start"], cfg["target"]["year_end"]

    files = find_variable_files(cfg["paths"]["raw_climate"], variable, y0, y1)
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 365})
    ds = standardise_coord_names(ds)
    ds = normalise_longitude(ds)

    da = ds[variable] if variable in ds.data_vars else ds[list(ds.data_vars)[0]]
    da = da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))

    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        lat_asc = bool(da.lat[0] < da.lat[-1])
        da = da.sel(
            lat=slice(lat_min, lat_max) if lat_asc else slice(lat_max, lat_min),
            lon=slice(lon_min, lon_max),
        )
        log.info("%s subset to bbox -> %s", variable, dict(da.sizes))

    da = UNIT_CONVERSIONS[spec.get("convert")](da)

    # Step 1: daily -> monthly, using the operation declared in config.yaml.
    op = spec["agg"]
    monthly = getattr(da.resample(time="1MS"), op)()
    # Step 2: monthly series -> one value per calendar month across the years.
    clim = monthly.groupby("time.month").mean("time")
    clim.name = variable
    clim.attrs["units"] = spec["out_unit"]
    clim.attrs["monthly_aggregation"] = op
    log.info("%s climatology computed (op=%s, unit=%s)", variable, op, spec["out_unit"])
    return clim.compute()


def extract_at_points(clim: xr.DataArray, points: pd.DataFrame,
                      resolution_deg: float) -> pd.DataFrame:
    """Sample a (month, lat, lon) climatology at the wheat locations."""
    lats, lons = clim["lat"].values, clim["lon"].values
    li, lj, valid = nearest_cell_index(points, lats, lons,
                                       max_distance_deg=resolution_deg)
    values = clim.values                                    # (12, nlat, nlon)
    out = pd.DataFrame(index=points.index)
    for m in range(12):
        col = f"{clim.name}_m{m + 1:02d}"
        v = values[m][li, lj].astype(float)
        out[col] = np.where(valid, v, np.nan)
    return out


def build_climate_features(cfg: dict, points: pd.DataFrame,
                           pad_deg: float = 1.0) -> pd.DataFrame:
    """Produce 7 variables x 12 months = 84 climate columns for each location."""
    bbox = (points["lat"].min() - pad_deg, points["lat"].max() + pad_deg,
            points["lon"].min() - pad_deg, points["lon"].max() + pad_deg)
    log.info("Climate bounding box: lat %.2f..%.2f  lon %.2f..%.2f", *bbox)

    frames = [points[["lat", "lon"]].reset_index(drop=True)]
    for variable in cfg["climate"]["variables"]:
        clim = monthly_climatology(cfg, variable, bbox=bbox)
        frames.append(
            extract_at_points(clim, points, cfg["climate"]["resolution_deg"])
            .reset_index(drop=True)
        )
    out = pd.concat(frames, axis=1)
    log.info("Climate feature table: %d rows x %d cols", *out.shape)
    from .grids import to_output_columns
    save_table(to_output_columns(out),
               Path(cfg["paths"]["interim"]) / cfg["output"]["climate_table"])
    return out
