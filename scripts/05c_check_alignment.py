import pandas as pd

def load(p):
    d = pd.read_csv(p).rename(columns={"lat": "latitude", "lon": "longitude"})
    d[["latitude", "longitude"]] = d[["latitude", "longitude"]].round(4)
    return d

cal  = load("data/interim/wheat_crop_calendar.csv")
clim = load("data/interim/climate_features_2010_2019.csv")
wf   = load("data/interim/wf_wheat_2010_2019.csv")

print("rows -> WF", len(wf), "| calendar", len(cal), "| climate", len(clim))

n = min(len(cal), len(clim))
a = cal[["latitude", "longitude"]].iloc[:n].reset_index(drop=True)
b = clim[["latitude", "longitude"]].iloc[:n].reset_index(drop=True)
same = (a == b).all(axis=1)
print("calendar vs climate, same position:", int(same.sum()), "of", n)
print("positional mismatches:", int((~same).sum()))

ck = set(zip(cal.latitude, cal.longitude))
lk = set(zip(clim.latitude, clim.longitude))
print("calendar coords missing from climate:", len(ck - lk))
print("climate coords not in calendar:", len(lk - ck))
print("duplicate coords -> calendar", int(cal.duplicated(['latitude','longitude']).sum()),
      "| climate", int(clim.duplicated(['latitude','longitude']).sum()))
print("climate tail:\n", clim[["latitude", "longitude"]].tail(6))