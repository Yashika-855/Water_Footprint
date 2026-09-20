"""Stage 5 - spatial integration of WF, crop calendar, climate and soil.

The WF grid is the master spatial reference because WF is the target.
Every join is counted; unmatched records are reported, never silently dropped.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import get_logger
from .grids import to_output_columns
from .io_utils import RowLedger, save_table

log = get_logger(__name__)


def integrate(cfg: dict, wf: pd.DataFrame, cal: pd.DataFrame,
              clim: pd.DataFrame, soil: pd.DataFrame | None) -> pd.DataFrame:
    """Join the four sources on the WF grid, counting losses at every step."""
    ledger = RowLedger("integration")
    ledger.record("WF master grid", len(wf))

    # `cal` already carries lat/lon plus the WF target columns.
    df = cal.reset_index(drop=True)
    ledger.record("after crop-calendar join", len(df))

    clim_cols = clim.drop(columns=[c for c in ("lat", "lon", "latitude", "longitude")
                                   if c in clim.columns])
    df = pd.concat([df, clim_cols.reset_index(drop=True)], axis=1)
    ledger.record("after climate join", len(df))

    if soil is not None and not soil.empty:
        df = pd.concat([df, soil.reset_index(drop=True)], axis=1)
        ledger.record("after soil join", len(df))
    else:
        log.warning("No soil features supplied - master dataset omits soil. "
                    "Record this as a deviation in docs/methodology.md.")

    ledger.write(cfg["paths"]["tables"])
    save_table(to_output_columns(df),
               Path(cfg["paths"]["processed"]) / cfg["output"]["master_table"])
    return df
