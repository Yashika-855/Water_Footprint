#!/usr/bin/env python
"""MODULE 7 - Step 1: what is in the aligned master, and what role does each column play?

Nothing is modified or saved here. This step only LOOKS, so that every later
decision (what is a feature, what is a target, what is bookkeeping) is made on
facts rather than assumptions.
"""
import re

import pandas as pd

PATH = "data/processed/master_wheat_dataset_aligned.csv"
CLIMATE_VARS = ["pr", "sfcWind", "tasmin", "tasmax", "hurs", "rsds", "ps"]
TARGETS = ["green_wf", "blue_wf", "total_wf"]

df = pd.read_csv(PATH)
print(f"Loaded {PATH}: {df.shape[0]} rows x {df.shape[1]} columns")
print("Missing values anywhere:", int(df.isna().sum().sum()))

# --- give every column exactly one role -------------------------------------------------
def role(col: str) -> str:
    if col in ("latitude", "longitude"):
        return "1 geographic"
    if col in TARGETS:
        return "2 target (y)"
    if col in ("planting_doy", "maturity_doy", "season_length_days", "n_growing_months"):
        return "3 calendar scalar"
    if re.fullmatch(r"cultdays_m\d\d", col):
        return "4 bookkeeping: cultivation days per month"
    if re.fullmatch(r"grow_m\d\d", col):
        return "5 bookkeeping: 7-day-rule flag per month"
    if col in ("wheat_type", "water_system", "irrigation"):
        return "6 label (constant?)"
    if any(re.fullmatch(rf"{v}_m\d\d", col) for v in CLIMATE_VARS):
        return "7 climate (var x month)"
    if col.startswith("soil_"):
        return "8 soil (one-hot)"
    return "9 UNCLASSIFIED"

roles = pd.Series({c: role(c) for c in df.columns})
print("\nColumns by role:")
print(roles.value_counts().sort_index().to_string())
assert not (roles == "9 UNCLASSIFIED").any(), "a column has no role - investigate"
assert roles.value_counts().sum() == df.shape[1]

# --- constant columns carry no information ----------------------------------------------
const = [c for c in df.columns if df[c].nunique() == 1]
print("\nColumns with a single value (cannot help a model):")
for c in const:
    print(f"   {c:14s} = {df[c].iloc[0]!r}")

# --- the calendar: seasons that wrap past New Year ----------------------------------------
wrap = df["maturity_doy"] < df["planting_doy"]
print(f"\nSeasons that cross New Year (maturity_doy < planting_doy): {int(wrap.sum())} "
      f"({wrap.mean():.1%} of rows)")
print("n_growing_months distribution (7-day rule already applied in Module 4):")
print(df["n_growing_months"].value_counts().sort_index().to_string())

# --- which calendar months are ever cultivated? -------------------------------------------
grow = df[[f"grow_m{m:02d}" for m in range(1, 13)]]
print("\nShare of cells cultivating wheat in each calendar month:")
print((grow.mean() * 100).round(1).rename(lambda c: c.replace("grow_", "")).to_string())