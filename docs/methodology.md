# Methodology and Reproducibility Record

Reproduction of EmeÃ§ et al. (2025), *High-resolution global modeling of wheat's
water footprint using a machine learning ensemble approach*, Ecological Processes.
Scope follows the implementation guide: no new algorithm, target, crop or
novelty. Grey water footprint is out of scope (the paper does not model it).

---

## 1. Data used

| Dataset | Role | Notes |
|---|---|---|
| Mialyk et al. ACEA unit water footprint (4TU, DOI 10.4121/7b45bcc6-686b-404d-a910-13c87156716a) | Targets: green, blue, total WF (mÂ³/t) | File used: `wf_unit_wheat_average_2010_2019.nc`. **Record the dataset version (v1/v2/v3) in `metadata/DATA_SOURCES.csv`.** |
| GSWP3-W5E5, ISIMIP3a `obsclim`, daily, 0.5Â° | 7 climate variables: pr, sfcWind, tasmin, tasmax, hurs, rsds, ps | Scenario `obsclim` (observed). `counterclim`, `spinclim` and `transclim` are synthetic and were not used. |
| GGCMI Phase 3 crop calendar (Zenodo 5062513) | Planting and maturity day | Four files: spring/winter wheat Ã— rainfed/irrigated. |
| FAO/UNESCO Digital Soil Map of the World (DSMW) | Soil | Attribute `DOMSOI` (dominant soil unit) only. |

---

## 2. Pipeline summary

1. WF target coarsened 5Ã—5 (â‰ˆ0.0833Â° â†’ â‰ˆ0.417Â°) and restricted to wheat cells.
2. Crop calendar attached by nearest 0.5Â° cell; seven-day rule applied.
3. Daily climate aggregated to monthly (precipitation summed, other variables
   averaged), then averaged across years to 12 values per variable.
4. Soil attached by polygon lookup (`geopandas.sjoin`).
5. All sources joined onto the WF grid (`05_build_master_dataset.py`).
6. Cleaning, 80:20 split, StandardScaler fitted on the training set only.
7. Pearson screening plus XGBoost importance â†’ top 20 features per target.
8. K = 5 K-means and Ward hierarchical clustering on the selected features.
9. One AdaBoostRegressor (decision-tree base learner) per cluster, per method,
   per target.
10. Evaluation on the held-out 20%: MAE, MSE, RMSE, MAPE, RÂ².

---

## 3. Counts compared with the paper

| Step | Paper | Ours |
|---|---|---|
| Initial geographic locations | 17,897 | 2,353 |
| After cleaning | 17,552 | 2,348 |
| Training (80%) | 14,042 | 1,878 |
| Testing (20%) | 3,510 | 470 |
| Candidate features | 102 | 171 |
| Selected features | 20 | 20 |
| Clusters (K) | 5 | 5 |

Our sample is about 13% of the paper's. The most likely cause is the 5Ã—5
coarsening producing fewer cells than the paper's grid, but **this has not been
verified**. To check, compare the grid dimensions of the raw WF file with the
count after coarsening.

---

## 4. Deviations from the paper

| # | Where | Paper | What we did | Why / consequence |
|---|---|---|---|---|
| 1 | Soil features | Not fully specified in the main text; paper mentions hydraulic conductivity, porosity and soil class | Used only `DOMSOI`, one-hot encoded into 81 columns | DSMW has no hydraulic properties. The 171 candidate features are 81 soil columns, 84 climate, 4 calendar, 2 geographic. |
| 2 | Non-cultivation months | Months with â‰¤ 7 cultivation days are excluded | Climate values in those months set to **0**, not NaN | NaN made every row incomplete (no location grows wheat all year) and cleaning removed all rows. A 0 can be confused with a genuine zero (e.g. no rainfall), so the flag columns `grow_mMM` were kept in the master table for reference. |
| 3 | Climate years | 2010â€“2019 | The `pr` file used is named `..._2011_2019.nc`, so climate covers **2011â€“2019**; all seven climate files checked: 2011-2019 | Only the 2011â€“2020 decade chunk was downloaded; the WF target is a 2010â€“2019 mean. |
| 4 | AdaBoost settings | Not published in full | sklearn defaults with tree depth 3, 50 estimators | Hyperparameter source is recorded in every model file. |
| 5 | Hierarchical test labels | Not described | Test rows assigned to the nearest training-cluster centroid | `AgglomerativeClustering` has no `predict()`. |
| 6 | Train/test split | 80:20, strategy not stated | Random 80:20, seed 42 | Neighbouring grid cells are spatially correlated, so test scores may be slightly optimistic. A spatial-block split would be a robustness check. |
| 7 | Soil coordinate system | n/a | DSMW shapefile has no CRS defined; assumed EPSG:4326 | 2,281 of 2,353 points matched; the 72 unmatched are likely coastal cells. |
| 8 | Monthly aggregation | Statistic not stated per variable | Sum for precipitation, mean for the rest | Declared in `config/config.yaml` under `climate.variables`. |
| 9 | Integration row ledger | n/a | Rows went 2,353 â†’ 2,348 â†’ 2,353 â†’ 2,353 | The âˆ’5 then +5 is unexplained; check for duplicated lat/lon rows. |

---

## 5. Results

Test set: 470 locations. MAPE is in percent. "Test score" in our output equals
RÂ²; the paper's "test score" is a different quantity (a percentage) and is not
comparable.

| Target | Method | MAE (mÂ³/t) | MAPE (%) | RÂ² |
|---|---|---|---|---|
| Green WF | K-means + AdaBoost | 275.0 | 29.3 | 0.354 |
| Green WF | Hierarchical + AdaBoost | 279.4 | 29.1 | 0.300 |
| Green WF | Global AdaBoost (baseline) | 489.3 | 62.8 | 0.189 |
| Blue WF | K-means + AdaBoost | 108.0 | 360.9 | 0.341 |
| Blue WF | Hierarchical + AdaBoost | 107.5 | 413.9 | 0.345 |
| Blue WF | Global AdaBoost (baseline) | 167.8 | 459.9 | âˆ’1.154 |
| Total WF | K-means + AdaBoost | 347.4 | 31.3 | 0.361 |
| Total WF | Hierarchical + AdaBoost | 352.4 | 31.9 | 0.424 |
| Total WF | Global AdaBoost (baseline) | 636.7 | 68.4 | 0.148 |

Paper reference (overall results, for comparison only): RÂ² 0.52 green, 0.40
blue, 0.69 total.

---

## 6. Research questions

| RQ | Finding |
|---|---|
| RQ1: can ML predict wheat WF? | Partly. RÂ² is 0.30â€“0.42 clustered, below the paper's 0.40â€“0.69. |
| RQ2: does clustering help? | Yes on this data. RÂ² roughly doubles for green and total, and MAE falls by 40â€“45%. For blue the global model has negative RÂ² (worse than predicting the mean) while clustered models reach about 0.34. |
| RQ3: K-means vs hierarchical | Close. Hierarchical is better on total and blue RÂ², K-means on green. With 470 test rows the differences should not be over-interpreted. |
| RQ4: accuracy per target | Total WF has the highest RÂ², blue the lowest MAE and the highest MAPE. |

**Blue WF MAPE.** Blue WF is near zero for many rainfed cells, so small absolute
errors become very large percentages. MAE and RÂ² are the more reliable measures
for blue.

---

## 7. Limitations

- The sample is about 13% of the paper's, with one soil attribute and default
  AdaBoost settings, which probably explains the lower RÂ².
- Climate may cover 2011â€“2019 rather than 2010â€“2019 (deviation 3).
- Zero-filling non-cultivation months (deviation 2) is a modelling choice not
  taken from the paper.
- The random split may be optimistic for spatial data.
- Results are from a single seed (42) with no repeated runs or confidence
  intervals.

---

## 8. Project rules observed

- No new algorithm, target variable or crop.
- No grey water footprint model.
- The 171-feature list is not claimed to match the paper's 102; it is written to
  `metadata/feature_dictionary.csv`.
- AdaBoost values are not presented as "paper settings".
- Unmatched or dropped records are counted in `results/metrics/row_ledger_*.csv`.
