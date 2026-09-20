# Methodology and Reproducibility Record

Fill this in **as you work**, not at the end. Every row here becomes a line in
the limitations section of your report.

---

## 1. Dataset versions actually used

See `metadata/DATA_SOURCES.csv`. If you could not establish which version the
paper used, say so explicitly and state which version you used instead.

---

## 2. Spatial convention (fix this before processing climate)

```
WF source ~0.0833 deg
        |  5 x 5 block aggregation (config: target.coarsen_factor)
        v
master grid ~0.417 deg
        |  nearest-centre lookup (config: climate.resolution_deg)
        v
climate grid 0.5 deg
```

Changing the master grid invalidates every downstream file. Decide once.

---

## 3. Climate aggregation

`config/config.yaml` → `climate.variables` is the single source of truth for
which monthly statistic is applied to each variable, and which unit conversion.
The guide is explicit that you must not apply one statistic blindly to all seven.

**Verify the NetCDF `units` attribute before trusting any conversion.**
`scripts/00_data_inventory.py` prints them for you.

| Variable | Monthly op | Conversion |
|---|---|---|
| `pr` | sum | kg m⁻² s⁻¹ × 86400 → mm/day |
| `tasmin`, `tasmax` | mean | K − 273.15 → °C |
| `hurs`, `rsds`, `ps`, `sfcWind` | mean | none |

---

## 4. Seven-day cultivation rule

Implemented in `src/crop_calendar.py::cultivation_days_per_month`. The growing
season is walked day by day from `planting_doy` to `maturity_doy`, wrapping past
day 365 for winter wheat. A month with **more than 7** cultivation days counts;
7 or fewer does not.

Climate values in non-cultivation months are set to NaN by
`src/features.py::mask_non_cultivation_months`, and columns that are empty
everywhere are dropped and logged.

Unit tests covering this (including the New Year wrap) are in
`tests/test_seven_day_rule.py`.

---

## 5. Known deviations from the paper

| # | Where | Paper | What we did | Why |
|---|---|---|---|---|
| 1 | Soil features | Not fully specified in main text | *(fill in)* | No machine-readable list published |
| 2 | AdaBoost hyperparameters | Not published in full | sklearn defaults | No supplementary values available |
| 3 | Hierarchical test-set labels | Not described | Nearest training-centroid assignment | `AgglomerativeClustering` has no `predict()` |
| 4 | Monthly climate aggregation | Stated per variable, statistic not given | See `config.yaml` | Documented choice, not blind reuse |
| 5 | Train/test split | 80:20, strategy unstated | Random 80:20 | See section 6 below |
| 6 | | | | |

---

## 6. Train/test split and spatial leakage

The pipeline defaults to a **random** 80:20 split (`preprocess.split_strategy`),
which is what the reported counts imply. But nearby grid cells are spatially
correlated, so a random split can make test performance look better than genuine
geographic generalisation. `src/preprocess.py` warns about this at runtime.

If you have time, run a spatial-block split as a robustness check and report
both numbers. That is a legitimate addition — it does not change the algorithm,
target, or crop.

---

## 7. Counts comparison

| Step | Paper | Ours |
|---|---|---|
| Initial geographic locations | 17,897 | |
| After incomplete-record removal | 17,552 | |
| Training (80%) | 14,042 | |
| Testing (20%) | 3,510 | |
| Candidate features | 102 | |
| Selected features | 20 | 20 |
| Clusters (K) | 5 | 5 |

Row-by-row loss at every join is written automatically to
`results/metrics/row_ledger_*.csv`. Paste the relevant ones here.

**Do not force your numbers to match the paper's.** Use them as sanity checks
and investigate large differences.

---

## 8. Answering the research questions

| RQ | What the pipeline gives you | What you still have to do |
|---|---|---|
| RQ1 — can ML predict WF? | `results/metrics/model_comparison.csv` | Interpret R² in context, not in isolation |
| RQ2 — does clustering help? | Clustered results only | **Train a single global AdaBoost as the baseline.** Without it you cannot claim clustering improves anything. |
| RQ3 — K-means vs hierarchical | Both, side by side | Note the centroid-assignment caveat (deviation 3) |
| RQ4 — accuracy per target | Run stages 06–09 with `--target all` | Compare green vs blue; blue WF is typically much harder |

RQ2 is the one most easily got wrong. The baseline is not optional.

---

## 9. Project rules (from the implementation guide)

- No new ML algorithm, target variable, crop, or research novelty without
  supervisor approval.
- **No grey water footprint model** — outside the base paper's scope.
- Do not claim you reproduced the exact 102-feature list unless verified against
  the supplementary material.
- Do not call invented hyperparameter values "paper settings".
- Do not silently drop unmatched records — record why they were removed.
