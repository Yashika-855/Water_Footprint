"""Files on disk use latitude/longitude; internals use lat/lon. Keep them in sync."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.grids import from_output_columns, to_output_columns


def test_round_trip():
    df = pd.DataFrame({"lat": [1.0], "lon": [2.0], "total_wf": [500.0]})
    out = to_output_columns(df)
    assert list(out.columns) == ["latitude", "longitude", "total_wf"]
    assert list(from_output_columns(out).columns) == ["lat", "lon", "total_wf"]


def test_config_targets_match_readme_names():
    from src.config import load_config
    cfg = load_config()
    assert set(cfg["target"]["variables"].values()) == {"green_wf", "blue_wf", "total_wf"}
