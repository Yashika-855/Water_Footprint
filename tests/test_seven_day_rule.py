"""The seven-day cultivation rule is the single most error-prone piece of
logic in this project (it wraps across New Year for winter wheat). Test it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.crop_calendar import cultivation_days_per_month


def test_simple_spring_season():
    # DOY 60 (1 Mar) to DOY 180 (29 Jun)
    days = cultivation_days_per_month(np.array([60.0]), np.array([180.0]))
    assert days.sum() == 121
    assert days[0, 0] == 0            # January: none
    assert days[0, 2] > 0             # March: yes
    assert days[0, 6] == 0            # July: none


def test_winter_season_wraps_new_year():
    # Planted 1 Oct (DOY 274), matures 15 Jul (DOY 196) the next year
    days = cultivation_days_per_month(np.array([274.0]), np.array([196.0]))
    assert days[0, 0] > 0             # January covered
    assert days[0, 9] > 0             # October covered
    assert days[0, 7] == 0            # August not covered
    assert days.sum() == (365 - 274 + 1) + 196


def test_nan_inputs_return_zero():
    days = cultivation_days_per_month(np.array([np.nan]), np.array([100.0]))
    assert days.sum() == 0


def test_threshold_excludes_short_months():
    # A 5-day season entirely inside one month must fail the > 7 day rule
    days = cultivation_days_per_month(np.array([100.0]), np.array([104.0]))
    assert (days > 7).sum() == 0
