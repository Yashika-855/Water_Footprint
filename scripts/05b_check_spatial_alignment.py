import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

wf = pd.read_csv(INTERIM / "wf_wheat_2010_2019.csv")
locations = pd.read_csv(INTERIM / "wheat_locations.csv")
calendar = pd.read_csv(INTERIM / "wheat_crop_calendar.csv")
climate = pd.read_csv(INTERIM / "climate_features_2010_2019.csv")


# ---------------------------------------------------------
# Standardize coordinate column names
# ---------------------------------------------------------

def standardize_coords(df):
    df = df.copy()

    if "lat" in df.columns and "latitude" not in df.columns:
        df = df.rename(columns={"lat": "latitude"})

    if "lon" in df.columns and "longitude" not in df.columns:
        df = df.rename(columns={"lon": "longitude"})

    return df


wf = standardize_coords(wf)
locations = standardize_coords(locations)
calendar = standardize_coords(calendar)
climate = standardize_coords(climate)


# ---------------------------------------------------------
# Basic information
# ---------------------------------------------------------

print("\n================ DATASET SHAPES ================\n")

print("WF:", wf.shape)
print("Locations:", locations.shape)
print("Calendar:", calendar.shape)
print("Climate:", climate.shape)


# ---------------------------------------------------------
# TEST 1
# WF coordinates vs wheat_locations coordinates
# ---------------------------------------------------------

print("\n================ TEST 1 ================\n")
print("WF vs wheat_locations coordinate alignment")

n = min(len(wf), len(locations))

wf_coords = wf[["latitude", "longitude"]].iloc[:n].reset_index(drop=True)
loc_coords = locations[["latitude", "longitude"]].iloc[:n].reset_index(drop=True)

matches = (
    wf_coords["latitude"].eq(loc_coords["latitude"])
    & wf_coords["longitude"].eq(loc_coords["longitude"])
)

print("Rows compared:", n)
print("Matching rows:", matches.sum())
print("Mismatching rows:", (~matches).sum())


# ---------------------------------------------------------
# TEST 2
# Calendar coordinates vs climate coordinates
# ---------------------------------------------------------

print("\n================ TEST 2 ================\n")
print("Calendar vs climate coordinate alignment")

n = min(len(calendar), len(climate))

cal_coords = calendar[["latitude", "longitude"]].iloc[:n].reset_index(drop=True)
clim_coords = climate[["latitude", "longitude"]].iloc[:n].reset_index(drop=True)

matches = (
    cal_coords["latitude"].eq(clim_coords["latitude"])
    & cal_coords["longitude"].eq(clim_coords["longitude"])
)

print("Rows compared:", n)
print("Matching rows:", matches.sum())
print("Mismatching rows:", (~matches).sum())

if (~matches).sum() > 0:
    print("\nFirst 10 mismatches:\n")

    mismatch_indices = matches[~matches].index[:10]

    for i in mismatch_indices:
        print(
            f"Row {i}: "
            f"calendar=({cal_coords.loc[i, 'latitude']}, "
            f"{cal_coords.loc[i, 'longitude']}), "
            f"climate=({clim_coords.loc[i, 'latitude']}, "
            f"{clim_coords.loc[i, 'longitude']})"
        )


# ---------------------------------------------------------
# TEST 3
# Are all calendar coordinates present in climate?
# ---------------------------------------------------------

print("\n================ TEST 3 ================\n")
print("Calendar coordinates missing from climate")

climate_coord_set = set(
    zip(climate["latitude"], climate["longitude"])
)

calendar_coord_set = set(
    zip(calendar["latitude"], calendar["longitude"])
)

missing_from_climate = calendar_coord_set - climate_coord_set

print("Calendar coordinate pairs:", len(calendar_coord_set))
print("Climate coordinate pairs:", len(climate_coord_set))
print("Calendar coordinates missing from climate:",
      len(missing_from_climate))

if missing_from_climate:
    print("\nExamples:")
    for coord in list(missing_from_climate)[:10]:
        print(coord)


# ---------------------------------------------------------
# TEST 4
# Which WF locations disappeared from calendar?
# ---------------------------------------------------------

print("\n================ TEST 4 ================\n")
print("WF locations absent from crop calendar")

location_coord_set = set(
    zip(locations["latitude"], locations["longitude"])
)

calendar_coord_set = set(
    zip(calendar["latitude"], calendar["longitude"])
)

missing_from_calendar = location_coord_set - calendar_coord_set

print("WF/location coordinate pairs:", len(location_coord_set))
print("Calendar coordinate pairs:", len(calendar_coord_set))
print("Locations absent from calendar:",
      len(missing_from_calendar))

if missing_from_calendar:
    print("\nLocations absent from calendar:")
    for coord in sorted(missing_from_calendar):
        print(coord)


print("\n================ DONE ================\n")