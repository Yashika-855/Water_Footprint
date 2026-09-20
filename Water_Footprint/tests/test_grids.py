import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from src.grids import (coarsen_grid, nearest_cell_index, normalise_longitude,
                        validate_coordinates)


def test_longitude_conversion():
    da = xr.DataArray(np.zeros((2, 4)),
                      coords={"lat": [0, 1], "lon": [0, 90, 180, 270]},
                      dims=["lat", "lon"])
    out = normalise_longitude(da)
    # 0, 90, 180, 270  ->  0, 90, -180, -90   (180 maps to -180 by convention)
    assert sorted(out.lon.values.tolist()) == [-180.0, -90.0, 0.0, 90.0]
    assert float(out.lon.max()) <= 180.0
    assert float(out.lon.min()) >= -180.0
    # coordinate must come back sorted ascending
    assert list(out.lon.values) == sorted(out.lon.values.tolist())


def test_coarsen_factor_five():
    da = xr.DataArray(np.ones((10, 10)),
                      coords={"lat": np.arange(10.0), "lon": np.arange(10.0)},
                      dims=["lat", "lon"], name="x")
    out = coarsen_grid(da, 5)
    assert out.shape == (2, 2)
    assert float(out.values[0, 0]) == 1.0


def test_nearest_cell_flags_far_points():
    pts = pd.DataFrame({"lat": [0.1, 50.0], "lon": [0.1, 50.0]})
    lats = np.array([0.0, 0.5, 1.0])
    lons = np.array([0.0, 0.5, 1.0])
    _, _, valid = nearest_cell_index(pts, lats, lons, max_distance_deg=0.5)
    assert valid[0] and not valid[1]


def test_validate_coordinates_drops_impossible():
    df = pd.DataFrame({"lat": [10.0, 200.0], "lon": [10.0, 10.0]})
    assert len(validate_coordinates(df)) == 1
