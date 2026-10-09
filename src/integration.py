"""Stage 5 - spatial integration of WF, crop calendar, climate and soil.

WHAT WAS WRONG IN THE OLD VERSION
---------------------------------
The old integrate() glued the tables together with

    pd.concat([df, clim_cols.reset_index(drop=True)], axis=1)

`concat(axis=1)` matches rows by POSITION (0, 1, 2, ...), not by location.
The calendar table had already lost 5 cells (no calendar within 0.5 deg), so
it had 2,348 rows, while climate and soil still had all 2,353. After the first
dropped cell, every row of the calendar table was glued to the climate/soil
row of a *different* grid cell, and 5 "ghost" rows with no coordinates were
appended at the bottom.

THE RULE THIS VERSION ENFORCES
------------------------------
* The WF grid is the master. Every other table is attached by its (lat, lon)
  KEY, never by position.
* Every join must be one-to-one (validate="one_to_one"), so a duplicated cell
  raises an error instead of silently multiplying rows.
* A join may never ADD rows. If the row count goes up, we raise.
* Cells that do not match are written to a file and recorded in the ledger,
  never dropped silently (guide, Module 6).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import get_logger
from .grids import to_output_columns
from .io_utils import RowLedger, save_table

log = get_logger(__name__)

KEY = ["_klat", "_klon"]
DECIMALS = 4          # grid spacing is ~0.4167 deg, so 4 dp can never merge two cells


def _add_key(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add integer join keys made from rounded lat/lon (float == float is unsafe)."""
    if not {"lat", "lon"}.issubset(df.columns):
        raise KeyError(
            f"The {name} table has no lat/lon columns, so it cannot be joined "
            f"by location. Columns found: {list(df.columns)[:8]}..."
        )
    out = df.copy()
    out["_klat"] = np.round(out["lat"].to_numpy() * 10**DECIMALS).astype("int64")
    out["_klon"] = np.round(out["lon"].to_numpy() * 10**DECIMALS).astype("int64")
    n_dup = int(out.duplicated(KEY).sum())
    if n_dup:
        raise ValueError(f"{name}: {n_dup} duplicated grid cells - join would be ambiguous.")
    return out


def _coords_from_wf_order(tbl: pd.DataFrame, wf: pd.DataFrame, name: str) -> pd.DataFrame:
    """Fallback for a table that carries NO coordinates (the soil table).

    build_soil_features() was run on wheat_locations.csv, i.e. row-for-row on the
    full WF table, so the *only* legitimate order-based alignment is against the
    FULL WF table (before the calendar dropped anything) and only if the lengths
    are identical. Anything else is refused.
    """
    if len(tbl) != len(wf):
        raise ValueError(
            f"{name} has {len(tbl)} rows and no coordinates, but the WF master grid "
            f"has {len(wf)}. Cannot align by order - rebuild {name} so it carries "
            f"lat/lon (see the soil.py patch)."
        )
    log.warning("%s table has no lat/lon; aligning to the FULL WF table by order. "
                "Patch src/soil.py so it writes lat/lon and this fallback is not needed.", name)
    out = tbl.reset_index(drop=True).copy()
    out.insert(0, "lon", wf["lon"].to_numpy())
    out.insert(0, "lat", wf["lat"].to_numpy())
    return out


def _checked_merge(left: pd.DataFrame, right: pd.DataFrame, name: str,
                   ledger: RowLedger) -> pd.DataFrame:
    """Left-join `right` onto `left` by key. Never adds rows, reports misses."""
    value_cols = [c for c in right.columns if c not in KEY + ["lat", "lon"]]
    before = len(left)
    merged = left.merge(right[KEY + value_cols], on=KEY, how="left",
                        validate="one_to_one", indicator=True)
    if len(merged) != before:                       # the old bug would have shown up here
        raise RuntimeError(f"Join with {name} changed the row count "
                           f"{before} -> {len(merged)}. Joins must never add rows.")
    n_miss = int((merged["_merge"] == "left_only").sum())
    merged = merged.drop(columns="_merge")
    ledger.record(f"after {name} join (keyed)", len(merged),
                  f"{n_miss} cells had no {name} row (kept as NaN, handled in Module 8)")
    return merged


def integrate(cfg: dict, wf: pd.DataFrame, cal: pd.DataFrame,
              clim: pd.DataFrame, soil: pd.DataFrame | None) -> pd.DataFrame:
    """Join the four sources on the WF grid, counting losses at every step."""
    ledger = RowLedger("integration")
    targets = list(cfg["target"]["variables"].values())

    master = _add_key(wf[["lat", "lon"] + targets], "WF")
    ledger.record("WF master grid", len(master))

    # ---- crop calendar: it was already filtered, so absent cells = no calendar ----
    cal_k = _add_key(cal, "calendar")
    cal_only = [c for c in cal_k.columns
                if c not in KEY + ["lat", "lon"] + targets]
    merged = master.merge(cal_k[KEY + cal_only], on=KEY, how="left",
                          validate="one_to_one", indicator=True)
    no_cal = merged[merged["_merge"] == "left_only"]
    if len(no_cal):
        out = Path(cfg["paths"]["tables"]) / "integration_unmatched_calendar.csv"
        no_cal[["lat", "lon"] + targets].to_csv(out, index=False)
        log.warning("%d WF cells have no crop calendar -> listed in %s", len(no_cal), out.name)
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge").reset_index(drop=True)
    ledger.record("after crop-calendar join (keyed)", len(merged),
                  f"{len(no_cal)} WF cells without a wheat calendar removed; "
                  f"listed in integration_unmatched_calendar.csv")
    df = merged

    # ---- climate: carries its own lat/lon, join by key ----
    df = _checked_merge(df, _add_key(clim, "climate"), "climate", ledger)

    # ---- soil: join by key if it has coordinates, else the guarded fallback ----
    if soil is not None and not soil.empty:
        soil_k = soil if {"lat", "lon"}.issubset(soil.columns) else \
            _coords_from_wf_order(soil, wf, "soil")
        df = _checked_merge(df, _add_key(soil_k, "soil"), "soil", ledger)
    else:
        log.warning("No soil features supplied - master dataset omits soil. "
                    "Record this as a deviation in docs/methodology.md.")

    df = df.drop(columns=KEY)

    # ---- hard post-conditions: fail loudly instead of writing a bad master ----
    assert df[["lat", "lon"] + targets].notna().all().all(), "NaN coordinate/target survived"
    assert not df.duplicated(["lat", "lon"]).any(), "duplicate grid cell in master"
    ledger.record("final integrated master", len(df))

    ledger.write(cfg["paths"]["tables"])
    save_table(to_output_columns(df),
               Path(cfg["paths"]["processed"]) / cfg["output"]["master_table"])
    return df