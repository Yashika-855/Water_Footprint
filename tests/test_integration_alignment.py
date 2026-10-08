"""Regression test for the Module 6 misalignment.

Each of the 6 grid cells carries its own ID as a marker in EVERY table
(green_wf = 100 + id, pr_m01 = id, soil one-hot column = id). After a correct
join, every row must show the same ID in all three places.
"""
import numpy as np
import pandas as pd
import pytest

from src.integration import integrate

N = 6
DROPPED = 2          # the calendar stage loses the cell with id 2 (a "mid-file" drop)


def _cfg(tmp_path):
    return {
        "target": {"variables": {"green": "green_wf", "blue": "blue_wf", "total": "total_wf"}},
        "paths": {"tables": tmp_path, "processed": tmp_path},
        "output": {"master_table": "master.csv"},
    }


def _tables():
    ids = np.arange(N)
    lat = 10 + ids * 0.4166667
    lon = 20 + ids * 0.4166667
    wf = pd.DataFrame({"lat": lat, "lon": lon, "green_wf": 100.0 + ids,
                       "blue_wf": 10.0 + ids, "total_wf": 110.0 + 2 * ids})
    cal = wf.drop(index=DROPPED).reset_index(drop=True)          # 5 rows left
    cal["planting_doy"] = 120.0
    clim = pd.DataFrame({"lat": lat, "lon": lon, "pr_m01": ids.astype(float)})
    soil = pd.DataFrame(np.eye(N, dtype=int), columns=[f"soil_DOMSOI_c{i}" for i in ids])
    return wf, cal, clim, soil


def test_old_positional_concat_is_misaligned():
    """Documents the bug: concat(axis=1) pairs row 2 of cal (id 3) with climate id 2."""
    wf, cal, clim, soil = _tables()
    glued = pd.concat([cal.reset_index(drop=True), clim[["pr_m01"]].reset_index(drop=True)], axis=1)
    wrong = (glued["green_wf"] - 100 != glued["pr_m01"]).sum()
    assert len(glued) == N                       # 5 + 1 ghost row, like the real master
    assert glued["green_wf"].isna().sum() == 1   # the ghost row with no WF value
    assert wrong >= 3                            # every row after the drop is mismatched


def test_keyed_integration_is_aligned(tmp_path):
    wf, cal, clim, soil = _tables()
    out = integrate(_cfg(tmp_path), wf, cal, clim, soil)
    assert len(out) == N - 1                                      # ghost row is gone
    assert out[["lat", "lon", "green_wf"]].notna().all().all()    # no NaN coordinates
    assert (out["green_wf"] - 100 == out["pr_m01"]).all()         # climate matches WF
    soil_id = out[[c for c in out.columns if c.startswith("soil_")]].to_numpy().argmax(axis=1)
    assert (soil_id == out["pr_m01"].to_numpy()).all()            # soil matches WF
    assert DROPPED not in out["pr_m01"].to_numpy()                # dropped cell really dropped


def test_soil_without_coordinates_and_wrong_length_is_refused(tmp_path):
    wf, cal, clim, soil = _tables()
    with pytest.raises(ValueError):
        integrate(_cfg(tmp_path), wf, cal, clim, soil.iloc[:-1])  # length != WF grid