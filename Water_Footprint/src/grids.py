"""Grid handling: longitude conventions, coarsening, nearest-cell matching.

The single most common source of silent error in this project is a longitude
convention mismatch (0..360 vs -180..180). Every loader funnels through
`normalise_longitude` before anything else happens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .config import get_logger

log = get_logger(__name__)


def normalise_longitude(ds: xr.Dataset | xr.DataArray,
                        lon_name: str = "lon") -> xr.Dataset | xr.DataArray:
    """Convert a 0..360 longitude axis to -180..180 and sort ascending."""
    if lon_name not in ds.coords:
        for alt in ("longitude", "x"):
            if alt in ds.coords:
                ds = ds.rename({alt: lon_name})
                break
    lon = ds[lon_name]
    if float(lon.max()) > 180.0:
        log.info("Longitude spans 0..360; converting to -180..180")
        ds = ds.assign_coords({lon_name: (((lon + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)
    return ds


def standardise_coord_names(ds: xr.Dataset) -> xr.Dataset:
    """Rename latitude/longitude variants to lat/lon."""
    rename = {}
    for cand, target in (("latitude", "lat"), ("y", "lat"),
                         ("longitude", "lon"), ("x", "lon")):
        if cand in ds.coords and target not in ds.coords:
            rename[cand] = target
    return ds.rename(rename) if rename else ds


def coarsen_grid(da: xr.DataArray, factor: int, how: str = "mean") -> xr.DataArray:
    """Block-aggregate a DataArray by `factor` in both lat and lon.

    5 arcmin (0.08333 deg) with factor=5 gives ~0.4167 deg, the paper's
    working resolution. Edge blocks are trimmed rather than partially filled.
    """
    coarse = da.coarsen(lat=factor, lon=factor, boundary="trim")
    if how == "mean":
        out = coarse.mean(skipna=True)
    elif how == "median":
        out = coarse.median(skipna=True)
    elif how == "sum":
        out = coarse.sum(skipna=True)
    else:
        raise ValueError(f"Unknown aggregation: {how}")
    log.info("Coarsened %s: %s -> %s", da.name, dict(da.sizes), dict(out.sizes))
    return out


def nearest_cell_index(points: pd.DataFrame, lats: np.ndarray, lons: np.ndarray,
                       max_distance_deg: float | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match each (lat, lon) point to the nearest source-grid cell centre.

    Returns (lat_idx, lon_idx, valid_mask). `valid_mask` is False where the
    nearest cell centre is further away than `max_distance_deg` in either axis,
    so you can count and report unmatched rows instead of silently accepting
    a bad join.
    """
    lat_idx = np.abs(lats[None, :] - points["lat"].to_numpy()[:, None]).argmin(axis=1)
    lon_idx = np.abs(lons[None, :] - points["lon"].to_numpy()[:, None]).argmin(axis=1)

    valid = np.ones(len(points), dtype=bool)
    if max_distance_deg is not None:
        dlat = np.abs(lats[lat_idx] - points["lat"].to_numpy())
        dlon = np.abs(lons[lon_idx] - points["lon"].to_numpy())
        valid = (dlat <= max_distance_deg) & (dlon <= max_distance_deg)
        log.info("Nearest-cell match: %d/%d within %.3f deg",
                 valid.sum(), len(valid), max_distance_deg)
    return lat_idx, lon_idx, valid


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop physically impossible coordinates; log how many."""
    before = len(df)
    ok = df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    if (~ok).any():
        log.warning("Dropping %d rows with impossible coordinates", int((~ok).sum()))
    return df.loc[ok].reset_index(drop=True) if before != ok.sum() else df


def to_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename internal lat/lon to the latitude/longitude names used in outputs.

    Internally the pipeline uses `lat`/`lon` because that matches the NetCDF
    coordinate names. Every file written to disk uses `latitude`/`longitude`,
    as documented in the repository README.
    """
    return df.rename(columns={"lat": "latitude", "lon": "longitude"})


def from_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse of `to_output_columns`, for reading files back in."""
    return df.rename(columns={"latitude": "lat", "longitude": "lon"})
