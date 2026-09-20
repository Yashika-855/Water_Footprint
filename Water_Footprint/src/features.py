"""Stage 6a - feature engineering and the candidate feature table.

Feature budget (documented, not guessed):
    geographic          2    lat, lon
    crop calendar       4    planting_doy, maturity_doy, season_length_days,
                             n_growing_months
    monthly climate    84    7 variables x 12 months
    soil               varies (whatever config soil.attributes yields)
    -------------------------------------------------------------------
    subtotal           90 + soil

The paper reports 102 initial features. Do NOT claim you reproduced their exact
list unless you verified it from the supplementary material - this module writes
the list you actually built to metadata/feature_dictionary.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import get_logger
from .io_utils import save_table

log = get_logger(__name__)

GEO_FEATURES = ["lat", "lon"]
CALENDAR_FEATURES = ["planting_doy", "maturity_doy", "season_length_days",
                     "n_growing_months"]


def mask_non_cultivation_months(df: pd.DataFrame,
                                climate_vars: list[str]) -> pd.DataFrame:
    """Apply the seven-day rule to the climate columns.

    Climate in a month where wheat is not cultivated is set to NaN, then the
    column is dropped if it is empty everywhere. Removed columns are recorded so
    the report can list them.
    """
    out = df.copy()
    removed: list[str] = []
    for var in climate_vars:
        for m in range(1, 13):
            col, flag = f"{var}_m{m:02d}", f"grow_m{m:02d}"
            if col not in out.columns or flag not in out.columns:
                continue
            out[col] = out[col].where(out[flag] == 1, np.nan)
            if out[col].isna().all():
                removed.append(col)
    if removed:
        out = out.drop(columns=removed)
        log.info("Dropped %d climate columns outside every cultivation period",
                 len(removed))
    out.attrs["removed_outside_season"] = removed
    return out


def build_feature_table(cfg: dict, df: pd.DataFrame,
                        apply_season_mask: bool = True
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (candidate feature table, feature dictionary)."""
    climate_vars = list(cfg["climate"]["variables"])
    if apply_season_mask:
        df = mask_non_cultivation_months(df, climate_vars)

    climate_cols = [c for c in df.columns
                    if any(c.startswith(f"{v}_m") for v in climate_vars)]
    soil_cols = [c for c in df.columns if c.startswith("soil_")]
    feature_cols = (GEO_FEATURES
                    + [c for c in CALENDAR_FEATURES if c in df.columns]
                    + climate_cols + soil_cols)

    targets = list(cfg["target"]["variables"].values())
    table = df[feature_cols + [t for t in targets if t in df.columns]].copy()

    def describe(col: str) -> tuple[str, str, str]:
        if col in GEO_FEATURES:
            return "geographic", "degrees", "Grid-cell centre coordinate"
        if col in CALENDAR_FEATURES:
            return "crop_calendar", "day / count", "GGCMI Phase 3 wheat calendar"
        if col in soil_cols:
            return "soil", "see DSMW key", "FAO/UNESCO DSMW attribute"
        if col in targets:
            return "target", cfg["target"]["unit"], "ACEA unit WF, 2010-2019 mean"
        var, month = col.rsplit("_m", 1)
        spec = cfg["climate"]["variables"].get(var, {})
        return ("climate", spec.get("out_unit", "?"),
                f"GSWP3-W5E5 {var}, month {int(month)}, "
                f"monthly op = {spec.get('agg', '?')}")

    rows = []
    for col in table.columns:
        group, unit, note = describe(col)
        rows.append({"feature": col, "group": group, "unit": unit,
                     "description": note,
                     "n_missing": int(table[col].isna().sum())})
    dictionary = pd.DataFrame(rows)

    log.info("Candidate features: %d (+%d targets) over %d rows",
             len(feature_cols), len(targets), len(table))
    log.info("Breakdown -> geo %d | calendar %d | climate %d | soil %d",
             len(GEO_FEATURES),
             len([c for c in CALENDAR_FEATURES if c in df.columns]),
             len(climate_cols), len(soil_cols))

    save_table(table, Path(cfg["paths"]["processed"]) / "features_full.csv")
    save_table(dictionary,
               Path(cfg["paths"]["metadata"]) / "feature_dictionary.csv")
    return table, dictionary
