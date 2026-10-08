#!/usr/bin/env python
"""Stage 5c - repair the row misalignment in master_wheat_dataset.csv.

THE PROBLEM
-----------
The master was built with a positional pd.concat(axis=1). The calendar table had
lost 5 cells, climate + soil had not, so from the first lost cell onward every
row carries the climate/soil of a DIFFERENT grid cell than its own WF/calendar
values, and 5 ghost rows (no coordinates) sit at the bottom.

THE IDEA (why we can repair it without the raw climate files)
-------------------------------------------------------------
Row p of the calendar block really belongs to original location (p + k(p)),
where k(p) = how many dropped cells came before it (0..5, never decreasing).
Soil is an independent witness: we can recompute the soil class at each row's
TRUE coordinates from the DSMW shapefile and compare it with the soil class
stored in the master at row p + k. The k(p) that makes them agree is the right
one. A monotone dynamic programme finds the 5 change points.

Usage
-----
    python scripts/05c_repair_alignment.py --check-only   # diagnose, write nothing
    python scripts/05c_repair_alignment.py                # diagnose + write repaired file

Output
------
    data/processed/master_wheat_dataset_aligned.csv       repaired master
    results/metrics/alignment_report.csv                  before/after evidence
    results/metrics/alignment_dropped_locations.csv       the cells that were lost
    results/metrics/alignment_uncertain_rows.csv          rows that need a human look
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from src.config import get_logger, load_config
from src.io_utils import RowLedger, save_table
from src.soil import find_soil_source

log = get_logger("05c_repair")
NA = "NA"                      # label for "no soil class" (all-zero one-hot / no polygon)
JUMP_PENALTY = 2.0             # cost of one extra dropped cell, vs 1.0 per soil mismatch
KNN = 8                        # neighbours for the climate-coherence check


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def split_columns(df: pd.DataFrame, cfg: dict):
    """Split master columns into (calendar/WF side, climate block, soil block)."""
    prefixes = tuple(f"{v}_m" for v in cfg["climate"]["variables"])
    clim = [c for c in df.columns if c.startswith(prefixes)]
    soil = [c for c in df.columns if c.startswith("soil_")]
    other = [c for c in df.columns if c not in set(clim) | set(soil)]
    return other, clim, soil


def soil_class_at(points: pd.DataFrame, cfg: dict) -> np.ndarray:
    """DSMW dominant-soil class at each (latitude, longitude) - same method as soil.py."""
    import geopandas as gpd
    attr = cfg["soil"]["attributes"][0]
    shp = gpd.read_file(find_soil_source(cfg["paths"]["raw_soil"]))[[attr, "geometry"]]
    if shp.crs is None:
        shp = shp.set_crs("EPSG:4326")
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(points["longitude"], points["latitude"]), crs="EPSG:4326")
    j = gpd.sjoin(pts, shp, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    return j[attr].fillna(NA).astype(str).to_numpy(object)


def stored_soil_class(block: pd.DataFrame, prefix: str) -> np.ndarray:
    """Which class does the one-hot block say? All-zero row -> NA (hidden missing!)."""
    names = np.array([c[len(prefix):] for c in block.columns], dtype=object)
    arr = block.to_numpy()
    return np.where(arr.sum(axis=1) > 0, names[arr.argmax(axis=1)], NA)


def fit_offsets(stored: np.ndarray, true: np.ndarray, max_off: int) -> np.ndarray:
    """k(p) for every calendar row p: non-decreasing, 0..max_off, minimum soil mismatches.

    cost[p, k] = 1 if the soil stored at master row (p + k) differs from the soil
    at calendar row p's true coordinates.
    """
    n, K = len(true), max_off + 1
    cost = np.ones((n, K))
    for k in range(K):
        cost[:, k] = stored[k:k + n] != true
    dp = np.full((n, K), np.inf)
    bp = np.zeros((n, K), int)
    dp[0] = cost[0] + JUMP_PENALTY * np.arange(K)
    for p in range(1, n):
        for k in range(K):
            cands = dp[p - 1, :k + 1] + JUMP_PENALTY * (k - np.arange(k + 1))
            bp[p, k] = int(cands.argmin())
            dp[p, k] = cands.min() + cost[p, k]
    k = int(dp[-1].argmin())
    path = [k]
    for p in range(n - 1, 0, -1):
        k = bp[p, k]
        path.append(k)
    return np.array(path[::-1]), cost


def offsets_from_changes(n: int, changes: list[int]) -> np.ndarray:
    k = np.zeros(n, int)
    for c in changes:
        k[c:] += 1
    return k


def coherence(coords: np.ndarray, Z: np.ndarray) -> float:
    """Mean distance between a row's climate and the mean climate of its 8 nearest
    neighbours. Climate varies smoothly in space, so a correct alignment scores LOW."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=KNN + 1).fit(coords)
    idx = nn.kneighbors(coords, return_distance=False)[:, 1:]
    return float(np.linalg.norm(Z - Z[idx].mean(axis=1), axis=1).mean())


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="diagnose only; write nothing")
    args = ap.parse_args()

    cfg = load_config()
    proc, tables = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["tables"])
    src_path = proc / cfg["output"]["master_table"]
    out_path = proc / "master_wheat_dataset_aligned.csv"
    ledger = RowLedger("alignment_repair")

    # ---- STEP 1: find the ghost rows ---------------------------------------------
    m = pd.read_csv(src_path)
    ledger.record("master as delivered", len(m))
    ghost = m["latitude"].isna()
    n_cal, n_ghost = int((~ghost).sum()), int(ghost.sum())
    log.info("STEP 1  %d rows with coordinates, %d ghost rows with NO coordinates", n_cal, n_ghost)
    if n_ghost == 0:
        log.info("Nothing to repair: no ghost rows. Master already aligned (or repaired).")
        return
    if not ghost.iloc[n_cal:].all():
        raise RuntimeError("Ghost rows are not all at the bottom - unexpected layout, stop.")

    other, clim_cols, soil_cols = split_columns(m, cfg)
    cal = m.iloc[:n_cal]
    prefix = f"soil_{cfg['soil']['attributes'][0]}_"

    # ---- STEP 2: soil witness ----------------------------------------------------
    true_cls = soil_class_at(cal[["latitude", "longitude"]].reset_index(drop=True), cfg)
    stored_cls = stored_soil_class(m[soil_cols], prefix)             # all n_cal + n_ghost rows
    naive_agree = float((stored_cls[:n_cal] == true_cls).mean())
    log.info("STEP 2  soil recomputed at each row's own coordinates. "
             "Naive agreement with the master's soil columns: %.1f%%", 100 * naive_agree)

    # ---- STEP 3: fit the offsets ---------------------------------------------------
    path, cost = fit_offsets(stored_cls, true_cls, n_ghost)
    changes = (np.where(np.diff(path) != 0)[0] + 1).tolist()
    # a jump of 2 at one row = two adjacent dropped cells; expand so we can shift them separately
    changes = sum(([c] * int(path[c] - path[c - 1]) for c in changes), [])
    fitted_agree = float((stored_cls[np.arange(n_cal) + path] == true_cls).mean())
    log.info("STEP 3  offsets k(p) fitted. Change rows: %s | final offset %d", changes, path[-1])
    log.info("        rows whose climate+soil belong to ANOTHER cell: %d (%.1f%%)",
             int((path > 0).sum()), 100 * (path > 0).mean())
    log.info("        soil agreement after alignment: %.1f%% (was %.1f%%)",
             100 * fitted_agree, 100 * naive_agree)

    # ---- STEP 4: settle +-1 row boundary ties with the climate witness ---------------
    clim_all = m[clim_cols].to_numpy(float)
    Z = (clim_all - clim_all.mean(0)) / np.where(clim_all.std(0) == 0, 1, clim_all.std(0))
    coords = cal[["latitude", "longitude"]].to_numpy(float)
    base_k = offsets_from_changes(n_cal, changes)
    base_A = Z[np.arange(n_cal) + base_k]
    from sklearn.neighbors import NearestNeighbors
    nbr = NearestNeighbors(n_neighbors=KNN + 1).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
    M = base_A[nbr].mean(axis=1)                                      # neighbour-mean climate

    def soil_cost(ch):
        kk = offsets_from_changes(n_cal, ch)
        return int((stored_cls[np.arange(n_cal) + kk] != true_cls).sum())

    best_cost, uncertain = soil_cost(changes), []
    for j in range(len(changes)):
        ties = []
        for d in range(-3, 4):
            trial = list(changes); trial[j] += d
            if trial != sorted(trial) or trial[j] < 1 or trial[j] >= n_cal:
                continue
            if soil_cost(trial) == best_cost:
                ties.append(trial[j])
        if len(ties) > 1:
            rows = np.arange(min(ties), max(ties))
            kb = offsets_from_changes(n_cal, changes[:j] + changes[j + 1:])   # offset without this change
            scores = {pos: sum(np.linalg.norm(Z[r + kb[r] + (1 if r >= pos else 0)] - M[r])
                               for r in rows) for pos in ties}
            choice = min(scores, key=scores.get)
            log.info("STEP 4  change %d: soil cannot distinguish rows %s; climate picks %d",
                     j + 1, ties, choice)
            changes[j] = choice
            uncertain += rows.tolist()
    k_final = offsets_from_changes(n_cal, changes)
    loc_idx = np.arange(n_cal) + k_final                               # master row each calendar row uses
    dropped = sorted(set(range(len(m))) - set(loc_idx.tolist()))
    assert len(dropped) == n_ghost and loc_idx.max() < len(m), "offsets inconsistent"

    # ---- STEP 5: independent proof - climate coherence before vs after -----------------
    before = coherence(coords, Z[:n_cal])
    after = coherence(coords, Z[loc_idx])
    log.info("STEP 5  climate-vs-neighbours distance (lower = more spatially coherent): "
             "before %.3f -> after %.3f", before, after)

    # ---- STEP 6: rebuild ------------------------------------------------------------------
    fixed = pd.concat(
        [cal[other].reset_index(drop=True),
         m.iloc[loc_idx][clim_cols + soil_cols].reset_index(drop=True)], axis=1)[m.columns]
    ledger.record("ghost rows removed, climate+soil re-attached by offset", len(fixed),
                  f"{n_ghost} unmatched locations dropped (positions {dropped})")

    no_soil = int((fixed[soil_cols].sum(axis=1) == 0).sum())
    log.info("STEP 6  repaired master: %d rows x %d cols | rows with ALL-ZERO soil "
             "one-hot (soil missing but invisible to dropna): %d", *fixed.shape, no_soil)

    report = pd.DataFrame([
        {"metric": "rows as delivered", "value": len(m)},
        {"metric": "ghost rows (no coordinates)", "value": n_ghost},
        {"metric": "rows with climate+soil from a different cell", "value": int((path > 0).sum())},
        {"metric": "soil agreement before", "value": round(naive_agree, 4)},
        {"metric": "soil agreement after", "value": round(fitted_agree, 4)},
        {"metric": "climate coherence before (lower is better)", "value": round(before, 4)},
        {"metric": "climate coherence after", "value": round(after, 4)},
        {"metric": "rows with all-zero soil one-hot after repair", "value": no_soil},
        {"metric": "change rows (calendar index)", "value": str(changes)},
    ])
    print("\n" + report.to_string(index=False))

    if args.check_only:
        log.info("--check-only: nothing written.")
        return

    save_table(fixed, out_path)
    save_table(report, tables / "alignment_report.csv")
    d = m.iloc[dropped][soil_cols]
    save_table(pd.DataFrame({"original_row": dropped,
                             "stored_soil_class": stored_soil_class(d, prefix),
                             "reason": "no wheat crop calendar within 0.5 deg"}),
               tables / "alignment_dropped_locations.csv")
    save_table(pd.DataFrame({"calendar_row": sorted(set(uncertain)),
                             "latitude": cal["latitude"].iloc[sorted(set(uncertain))].to_numpy(),
                             "longitude": cal["longitude"].iloc[sorted(set(uncertain))].to_numpy(),
                             "note": "boundary tie - neighbouring cells share soil class; "
                                     "climate may belong to the adjacent cell"}),
               tables / "alignment_uncertain_rows.csv")
    ledger.write(tables)
    log.info("Wrote %s. Original master left untouched.", out_path.name)


if __name__ == "__main__":
    main()