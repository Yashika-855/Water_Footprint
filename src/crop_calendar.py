"""MODULE 4 - wheat crop calendar and the paper's seven-day rule.

Source: GGCMI Phase 3 crop calendar (Jaegermeyr et al. 2021), Zenodo 5062513.
        0.5 deg, planting day + maturity day, separate spring (swh) and
        winter (wwh) wheat, rainfed (noirr / rf) and irrigated (firr / ir).

Output: data/interim/wheat_crop_calendar.csv plus a boolean cultivation mask
        `grow_m01 ... grow_m12` per location.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import get_logger
from .grids import nearest_cell_index, normalise_longitude, standardise_coord_names
from .io_utils import RowLedger, save_table

log = get_logger(__name__)

DAYS_IN_MONTH = np.array([calendar.monthrange(2001, m)[1] for m in range(1, 13)])
MONTH_START = np.concatenate([[1], np.cumsum(DAYS_IN_MONTH)[:-1] + 1])   # DOY of 1st
MONTH_END = np.cumsum(DAYS_IN_MONTH)                                     # DOY of last


def find_calendar_file(raw_dir: str | Path, wheat_type: str, irrig: str) -> Path:
    """Find e.g. the spring-wheat rainfed calendar file."""
    raw_dir = Path(raw_dir)
    aliases = {"noirr": ["noirr", "rf", "rainfed"], "firr": ["firr", "ir", "irrigated"]}
    for p in sorted(raw_dir.rglob("*.nc*")):
        n = p.name.lower()
        if wheat_type in n and any(a in n for a in aliases[irrig]):
            return p
    raise FileNotFoundError(
        f"No calendar file for {wheat_type}/{irrig} in {raw_dir}. "
        f"Present: {[p.name for p in raw_dir.rglob('*.nc*')]}"
    )


def cultivation_days_per_month(planting: np.ndarray, maturity: np.ndarray) -> np.ndarray:
    """Days of the growing season falling in each calendar month.

    Handles seasons that wrap across New Year (winter wheat) by walking the
    day-of-year axis twice. Returns an (n_points, 12) integer array.
    """
    n = len(planting)
    out = np.zeros((n, 12), dtype=np.int16)

    valid = np.isfinite(planting) & np.isfinite(maturity)
    p = np.where(valid, planting, 1).astype(int)
    m = np.where(valid, maturity, 1).astype(int)

    # Season length, wrapping if maturity precedes planting.
    length = np.where(m >= p, m - p + 1, (365 - p + 1) + m)

    for i in range(n):
        if not valid[i]:
            continue
        doys = ((np.arange(length[i]) + p[i] - 1) % 365) + 1
        months = np.searchsorted(MONTH_END, doys, side="left")     # 0-based month
        counts = np.bincount(months, minlength=12)
        out[i] = counts[:12]
    return out


def load_calendar(cfg: dict, wheat_type: str = "swh", irrig: str = "noirr"
                  ) -> xr.Dataset:
    path = find_calendar_file(cfg["paths"]["raw_calendar"], wheat_type, irrig)
    ds = xr.open_dataset(path)
    ds = standardise_coord_names(ds)
    ds = normalise_longitude(ds)
    log.info("Loaded calendar %s (vars: %s)", path.name, list(ds.data_vars))
    return ds


def attach_calendar(points: pd.DataFrame, cfg: dict,
                    wheat_type: str = "swh", irrig: str = "noirr") -> pd.DataFrame:
    """Attach planting/maturity day and monthly cultivation flags to WF points."""
    ccfg = cfg["crop_calendar"]
    ledger = RowLedger(f"calendar_{wheat_type}_{irrig}")
    ledger.record("input WF locations", len(points))

    ds = load_calendar(cfg, wheat_type, irrig)
    plant_var = next(v for v in ds.data_vars if "plant" in v.lower())
    mat_var = next(v for v in ds.data_vars
                   if "matur" in v.lower() or "harvest" in v.lower())

    lats, lons = ds["lat"].values, ds["lon"].values
    li, lj, valid = nearest_cell_index(points, lats, lons,
                                       max_distance_deg=ccfg["resolution_deg"])

    plant = ds[plant_var].values[li, lj].astype(float)
    mat = ds[mat_var].values[li, lj].astype(float)

    out = points.copy()
    out["planting_doy"] = np.where(valid, plant, np.nan)
    out["maturity_doy"] = np.where(valid, mat, np.nan)
    out["wheat_type"] = wheat_type
    out["water_system"] = "rainfed" if irrig == "noirr" else "irrigated"
    out["irrigation"] = irrig

    days = cultivation_days_per_month(out["planting_doy"].to_numpy(),
                                      out["maturity_doy"].to_numpy())
    for m in range(12):
        out[f"cultdays_m{m + 1:02d}"] = days[:, m]

    # ---- Paper rule: <= 7 cultivation days means the month does not count ----
    thr = ccfg["min_cultivation_days"]
    for m in range(12):
        out[f"grow_m{m + 1:02d}"] = (days[:, m] > thr).astype(int)
    out["n_growing_months"] = out[[f"grow_m{m:02d}" for m in range(1, 13)]].sum(axis=1)
    out["season_length_days"] = days.sum(axis=1)

    log.info("Seven-day rule applied (threshold > %d days). "
             "Mean growing months per cell: %.2f", thr, out["n_growing_months"].mean())

    before = len(out)
    out = out.dropna(subset=["planting_doy", "maturity_doy"])
    ledger.record("after dropping cells without a calendar", len(out),
                  f"{before - len(out)} unmatched")
    out = out[out["n_growing_months"] > 0].reset_index(drop=True)
    ledger.record("after requiring >=1 growing month", len(out))
    ledger.write(cfg["paths"]["tables"])

    from .grids import to_output_columns
    save_table(to_output_columns(out),
               Path(cfg["paths"]["interim"]) / cfg["output"]["calendar_table"])
    return out
