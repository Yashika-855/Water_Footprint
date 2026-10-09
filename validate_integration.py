import os
import pandas as pd

CALENDAR_PATH = "data/interim/wheat_crop_calendar.csv"
CLIMATE_PATH = "data/interim/climate_features_2010_2019.csv"
WF_PATH = "data/interim/wf_wheat_2010_2019.csv"
LOCATIONS_PATH = "data/interim/wheat_locations.csv"


def load(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
    df[["latitude", "longitude"]] = df[["latitude", "longitude"]].round(4)
    return df


calendar, climate = load(CALENDAR_PATH), load(CLIMATE_PATH)
wf, locations = load(WF_PATH), load(LOCATIONS_PATH)

print("=" * 60, "\nSOURCE TABLE SIZES\n" + "=" * 60)
for name, d in [("WF", wf), ("Locations", locations),
                ("Calendar", calendar), ("Climate", climate)]:
    print(f"{name:10}", d.shape,
          "| duplicate coords:", int(d.duplicated(["latitude", "longitude"]).sum()))

# TEST 1: WF vs locations (positional)
n1 = min(len(wf), len(locations))
m1 = (wf[["latitude", "longitude"]].iloc[:n1].reset_index(drop=True)
      == locations[["latitude", "longitude"]].iloc[:n1].reset_index(drop=True)).all(axis=1)
print("\n" + "=" * 60, "\nTEST 1 - WF vs wheat_locations\n" + "=" * 60)
print("Matching rows:", int(m1.sum()), "of", n1)

# TEST 2: calendar vs climate (positional)
n = min(len(calendar), len(climate))
cal_c = calendar[["latitude", "longitude"]].iloc[:n].reset_index(drop=True)
cli_c = climate[["latitude", "longitude"]].iloc[:n].reset_index(drop=True)
pm = (cal_c == cli_c).all(axis=1)
print("\n" + "=" * 60, "\nTEST 2 - Calendar vs Climate positional alignment\n" + "=" * 60)
print("Rows compared:", n)
print("Matching positions:", int(pm.sum()))
print("Mismatching positions:", int((~pm).sum()))
for i in pm[~pm].index[:10]:
    print(f"Row {i}: calendar={tuple(cal_c.loc[i])} climate={tuple(cli_c.loc[i])}")

# TEST 3: membership
cal_k = set(zip(calendar.latitude, calendar.longitude))
cli_k = set(zip(climate.latitude, climate.longitude))
print("\n" + "=" * 60, "\nTEST 3 - Coordinate membership\n" + "=" * 60)
print("Unique calendar coords:", len(cal_k))
print("Unique climate coords:", len(cli_k))
print("Calendar coords missing from climate:", len(cal_k - cli_k))
extra = sorted(cli_k - cal_k)
print("Climate coords not in calendar:", len(extra))

os.makedi