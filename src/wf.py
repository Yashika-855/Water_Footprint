"""Stage 1 - build the wheat water-footprint target table.

Source: Mialyk et al. ACEA outputs, 4TU.ResearchData
        DOI 10.4121/7b45bcc6-686b-404d-a910-13c87156716a
File:   unit_wf_selected_crops_average_2010_2019.zip  (unit WF in m3/t,
        already averaged over 2010-2019 -> no annual averaging needed)

Output: data/interim/wf_wheat_2010_2019.csv
        latitude | longitude | green_wf | blue_wf | total_wf
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from .config import get_logger
from .grids import (coarsen_grid, normalise_longitude, standardise_coord_names,
                    to_output_columns, validate_coordinates)
from .io_utils import RowLedger, save_table

log = get_logger(__name__)

WF_VARIABLE_CANDIDATES = {
    "green": ["wf_unit_rainfed_green", "wf_unit_irrigated_green"],
    "blue":  ["wf_unit_rainfed_blue", "wf_unit_irrigated_blue"],
}


def find_wheat_files(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    files = sorted(p for p in raw_dir.rglob("*.nc*")
                   if "wheat" in p.name.lower() or "_whe" in p.name.lower())
    if not files:
        raise FileNotFoundError(
            f"No wheat NetCDF found under {raw_dir}. Unzip "
            f"unit_wf_selected_crops_average_2010_2019.zip there. "
            f"Files present: {[p.name for p in raw_dir.rglob('*')][:20]}"
        )
    log.info("Wheat WF files found: %s", [f.name for f in files])
    return files


def _has_rainfed_irrigated_split(ds: xr.Dataset) -> bool:
    return all(v in ds.data_vars for v in WF_VARIABLE_CANDIDATES["green"])


def _resolve_variable(ds: xr.Dataset, component: str) -> str:
    for cand in WF_VARIABLE_CANDIDATES[component]:
        if cand in ds.data_vars:
            return cand
    raise KeyError(
        f"Could not find the '{component}' WF variable. Variables in file: "
        f"{list(ds.data_vars)}. Edit WF_VARIABLE_CANDIDATES in src/wf.py "
        f"after reading the provider readme.pdf."
    )


def build_target(cfg: dict) -> pd.DataFrame:
    tcfg = cfg["target"]
    g, b, t = (tcfg["variables"]["green"], tcfg["variables"]["blue"],
               tcfg["variables"]["total"])
    ledger = RowLedger("wf")

    files = find_wheat_files(cfg["paths"]["raw_wf"])
    ds = (xr.open_mfdataset(files, combine="by_coords") if len(files) > 1
          else xr.open_dataset(files[0]))
    ds = normalise_longitude(standardise_coord_names(ds))

    if "time" in ds.dims:
        ds = ds.sel(time=slice(str(tcfg["year_start"]), str(tcfg["year_end"])))
        log.info("Averaging %d time steps over %s-%s",
                 ds.sizes["time"], tcfg["year_start"], tcfg["year_end"])
        ds = ds.mean("time", skipna=True)

    if _has_rainfed_irrigated_split(ds):
        log.warning(
            "WF file gives rainfed/irrigated values separately, not a single "
            "green/blue value. Averaging whichever water system(s) have data "
            "per cell (skipna). Record this as a deviation in docs/methodology.md."
        )
        green = xr.concat(
            [ds["wf_unit_rainfed_green"], ds["wf_unit_irrigated_green"]],
            dim="water_system"
        ).mean("water_system", skipna=True)
        blue = xr.concat(
            [ds["wf_unit_rainfed_blue"], ds["wf_unit_irrigated_blue"]],
            dim="water_system"
        ).mean("water_system", skipna=True)
    else:
        green = ds[_resolve_variable(ds, "green")]
        blue = ds[_resolve_variable(ds, "blue")]

    factor, how = tcfg["coarsen_factor"], tcfg["aggregation"]
    stacked = xr.Dataset({g: coarsen_grid(green, factor, how),
                          b: coarsen_grid(blue, factor, how)})
    df = stacked.to_dataframe().reset_index()
    ledger.record("cells after 5x5 coarsening", len(df))

    df = df[["lat", "lon", g, b]].dropna(subset=[g, b])
    ledger.record("cells with green+blue present", len(df),
                  "ocean / non-wheat cells removed")

    df[t] = df[g] + df[b]
    df = df[df[t] > 0]
    ledger.record(f"cells with {t} > 0", len(df))

    df = validate_coordinates(df).reset_index(drop=True)
    ledger.record("final wheat locations", len(df))

    ref = cfg["preprocess"]["paper_reference"]["initial_locations"]
    log.info("Sanity check: paper reports ~%d locations, you have %d. "
             "Investigate a large difference; do not force the number.", ref, len(df))
    ledger.write(cfg["paths"]["tables"])

    out_dir = Path(cfg["paths"]["interim"])
    save_table(to_output_columns(df), out_dir / cfg["output"]["wf_table"])
    save_table(to_output_columns(df[["lat", "lon"]]),
               out_dir / cfg["output"]["locations_table"])
    return df
