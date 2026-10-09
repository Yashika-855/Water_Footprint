#!/usr/bin/env python
"""MODULE 7 - Step 2: the seven-day rule and the "month is not cultivated" problem.

The paper only lets a month's climate count if wheat grows there for MORE than
7 days. grow_mXX (from Module 4) already holds that yes/no per cell and month.
The open question: what value does a climate column take when grow_mXX == 0?

This script MEASURES the options. It writes nothing.
"""
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

PATH = "data/processed/master_wheat_dataset_aligned.csv"
VARS = ["pr", "sfcWind", "tasmin", "tasmax", "hurs", "rsds", "ps"]
df = pd.read_csv(PATH)
months = [f"{m:02d}" for m in range(1, 13)]
clim_cols = [f"{v}_m{m}" for v in VARS for m in months]
grow = df[[f"grow_m{m}" for m in months]].to_numpy().astype(bool)         # (rows, 12)

# ---------------------------------------------------------------- A. how much is masked?
masked_share = 1 - grow.mean()
print(f"A. Share of (cell, month) climate values that fall OUTSIDE the cultivation period: "
      f"{masked_share:.1%}")
print(f"   Rows that would have at least one masked month: {(~grow).any(axis=1).mean():.0%}")

# ---------------------------------------------------------------- B. option 'NaN'
print("\nB. Option NaN (leave non-cultivated months empty):")
nan_df = df[clim_cols].copy()
for v in VARS:
    for i, m in enumerate(months):
        nan_df.loc[~grow[:, i], f"{v}_m{m}"] = np.nan
print(f"   complete rows left after dropna(): {len(nan_df.dropna())} of {len(df)}")
print("   -> Module 8 would delete the whole dataset. NaN cannot be the stored value.")

# ---------------------------------------------------------------- C. option 'zero fill' (current features.py)
print("\nC. Option zero-fill (what src/features.py does today).")
print("   How far is a fake 0 from the REAL in-season values? (in standard deviations)")
rows = []
for v in VARS:
    gaps = []
    for i, m in enumerate(months):
        real = df.loc[grow[:, i], f"{v}_m{m}"]
        if real.std() > 0:
            gaps.append(abs(0 - real.mean()) / real.std())
    rows.append((v, float(np.mean(gaps))))
print(pd.DataFrame(rows, columns=["variable", "avg distance of 0 from real values (std)"])
      .round(1).to_string(index=False))

# ---------------------------------------------------------------- D. does the choice change a model?
print("\nD. Does it matter for a model? 5-fold CV R2 for green_wf (AdaBoost, depth-3 trees).")
base = df[["latitude", "longitude", "planting_doy", "maturity_doy",
           "season_length_days", "n_growing_months"]]


def make(fill):
    X = df[clim_cols].copy()
    for v in VARS:
        for i, m in enumerate(months):
            col = f"{v}_m{m}"
            if fill == "raw":
                continue
            inside = X.loc[grow[:, i], col]
            X.loc[~grow[:, i], col] = 0.0 if fill == "zero" else inside.mean()
    return pd.concat([base, X], axis=1)


cv = KFold(5, shuffle=True, random_state=42)
model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=50, random_state=42)
for fill, label in (("zero", "zero-fill"), ("mean", "fill with in-season mean"),
                    ("raw", "no masking (all 12 raw months)")):
    r2 = cross_val_score(model, make(fill), df["green_wf"], cv=cv, scoring="r2")
    print(f"   {label:34s} R2 = {r2.mean():.3f} (+/- {r2.std():.3f})")