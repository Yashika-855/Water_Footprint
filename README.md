# Wheat Water Footprint ML

End-to-end research project for estimating **wheat green, blue and total water footprint (WF)** from global environmental data using:

- Water-footprint target data
- Crop calendars
- Climate variables
- Soil variables
- Feature selection
- K-means and hierarchical clustering
- Cluster-wise AdaBoost regression
- MAE, MSE, MAPE and R² evaluation

> **Important:** This repository is a research implementation scaffold. Do not claim exact reproduction of a paper until the paper version, feature list, preprocessing rules, spatial aggregation and train/test protocol have been verified against the paper and supplementary material.

---

## 1. Project pipeline

```text
Global WF + Climate + Soil + Crop Calendar
                    |
                    v
        Data integration / cleaning
                    |
                    v
             Master wheat grid
                    |
                    v
          Feature selection 102 -> 20
                    |
                    v
             Clustering K = 5
              /            \
         K-means       Hierarchical
              \            /
               \          /
                v        v
          Cluster-wise AdaBoost
                    |
                    v
       Green / Blue / Total WF
                    |
                    v
       MAE / MSE / MAPE / R2
```

---

## 2. Data sources

Keep downloaded source files outside Git. Record every source in `metadata/DATA_SOURCES.csv`.

### Water footprint

4TU.ResearchData, DOI:

`10.4121/7b45bcc6-686b-404d-a910-13c87156716a`

Primary target file:

`unit_wf_selected_crops_average_2010_2019.zip`

Expected information:

- wheat cells
- green WF
- blue WF
- unit WF in m3/t
- 2010–2019 average
- NetCDF/CSV material

Before processing, read the provider README and verify that wheat is included.

### Crop calendar

Zenodo record:

`5062513`

Required wheat combinations:

- spring wheat + rainfed
- spring wheat + irrigated
- winter wheat + rainfed
- winter wheat + irrigated

Calendar variables are expected to include planting and maturity day-of-year values.

### Climate

ISIMIP3a / GSWP3-W5E5:

`ISIMIP3a/InputData/climate/atmosphere/obsclim/global/daily/historical/GSWP3-W5E5`

Required variables:

- `hurs`
- `sfcWind`
- `tasmax`
- `tasmin`
- `rsds`
- `ps`
- `pr`

For 2010–2019, use the relevant decade chunks from the repository and subset before doing expensive processing.

### Soil

Candidate source:

- FAO/UNESCO Digital Soil Map of the World
- HWSD v1.2 if required by the paper's actual feature definitions

Do not invent the paper's soil feature list. Verify Table 1 and supplementary material first.

---

## 3. Recommended repository structure

```text
wheat-water-footprint-ml/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
│
├── config/
│   └── config.yaml
│
├── metadata/
│   ├── DATA_SOURCES.csv
│   └── feature_dictionary.csv
│
├── data/
│   ├── raw/
│   │   ├── wf/
│   │   ├── crop_calendar/
│   │   ├── climate/
│   │   └── soil/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_wf_exploration.ipynb
│   ├── 02_crop_calendar_exploration.ipynb
│   ├── 03_climate_exploration.ipynb
│   ├── 04_soil_exploration.ipynb
│   └── 05_model_results.ipynb
│
├── src/
│   ├── __init__.py
│   ├── io_utils.py
│   ├── wf.py
│   ├── crop_calendar.py
│   ├── climate.py
│   ├── soil.py
│   ├── integration.py
│   ├── features.py
│   ├── clustering.py
│   ├── models.py
│   └── evaluation.py
│
├── scripts/
│   ├── 01_prepare_wf.py
│   ├── 02_prepare_calendar.py
│   ├── 03_prepare_climate.py
│   ├── 04_prepare_soil.py
│   ├── 05_build_master_dataset.py
│   ├── 06_select_features.py
│   ├── 07_cluster.py
│   ├── 08_train_models.py
│   └── 09_evaluate.py
│
├── models/
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
│
└── docs/
    ├── methodology.md
    └── data_collection.md
```

---

## 4. Installation

### Option A: conda

```bash
conda env create -f environment.yml
conda activate wheat-wf-ml
```

### Option B: pip

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

For large NetCDF processing, install CDO separately.

Check:

```bash
cdo -V
```

---

## 5. Data workflow

### Stage 1 — Water footprint

Put the downloaded WF archive under:

```text
data/raw/wf/
```

Run:

```bash
python scripts/01_prepare_wf.py
```

Expected output:

```text
data/interim/wf_wheat_2010_2019.csv
```

Minimum target columns should eventually look like:

```text
latitude
longitude
green_wf
blue_wf
total_wf
```

The exact variable names must be adjusted after inspecting the provider files.

---

### Stage 2 — Crop calendar

Put the four wheat calendar files under:

```text
data/raw/crop_calendar/
```

Run:

```bash
python scripts/02_prepare_calendar.py
```

Expected output:

```text
data/interim/wheat_crop_calendar.csv
```

Recommended fields:

```text
latitude
longitude
wheat_type
water_system
planting_doy
maturity_doy
```

---

### Stage 3 — Climate

Do **not** download/process the whole global climate archive if you only need wheat locations.

First create:

```text
data/interim/wheat_locations.csv
```

with:

```text
latitude
longitude
```

Then subset climate data to those locations.

Required climate variables:

```text
hurs
sfcWind
tasmax
tasmin
rsds
ps
pr
```

Run:

```bash
python scripts/03_prepare_climate.py
```

Expected output:

```text
data/interim/climate_features_2010_2019.csv
```

---

## 6. Climate unit conversions

Check the actual metadata of each downloaded file before conversion.

Common conversions include:

### Temperature

If stored in Kelvin:

```text
Celsius = Kelvin - 273.15
```

### Precipitation

If stored as kg m-2 s-1:

```text
mm/day = kg m-2 s-1 * 86400
```

### Surface pressure

If stored in Pa:

```text
kPa = Pa / 1000
```

Do not blindly apply these conversions. Verify the NetCDF variable attributes first.

---

## 7. Seven-day cultivation rule

The crop calendar provides planting and maturity day-of-year.

The project should derive climate statistics for the relevant crop-growth period according to the paper's seven-day rule.

For each wheat location:

```text
planting DOY
      |
      v
growth-period window
      |
      v
daily climate observations
      |
      v
monthly / seasonal aggregation
      |
      v
climate features
```

The exact interpretation of the seven-day rule must be implemented from the paper's Section 7.1 rather than guessed.

---

## 8. Master dataset

Run:

```bash
python scripts/05_build_master_dataset.py
```

This creates:

```text
data/processed/master_wheat_dataset.csv
```

Conceptually:

```text
latitude
longitude

green_wf
blue_wf
total_wf

planting_doy
maturity_doy
wheat_type
water_system

climate features...
soil features...
```

The master dataset is the main input to the ML pipeline.

---

## 9. Restrict to wheat cells early

The WF source may contain millions of global cells.

Do this as early as possible:

```text
Global WF grid
      |
      v
Keep wheat cells
      |
      v
~17,900 locations (if this matches the paper)
```

Do not hard-code `17,900` as a required number. Use it as a sanity check and investigate large differences.

This location list should then drive the climate subsetting.

---

## 10. Spatial resolution

The workflow described in the project guide uses:

```text
WF source: ~0.0833°
        |
        | 5 x 5 aggregation
        v
~0.417° master grid
        |
        | nearest-center climate lookup
        v
0.5° climate grid
```

Fix this spatial convention before processing climate data.

Avoid repeatedly changing the master grid because every downstream dataset depends on it.

---

## 11. Feature selection

After integration, the paper's workflow expects approximately:

```text
102 candidate features
        |
        v
Feature selection
        |
        v
20 selected features
```

Run:

```bash
python scripts/06_select_features.py
```

The exact feature-selection algorithm must be matched to the paper.

The script currently separates the **feature-selection stage** from the downstream model so you can replace the method after verifying the paper.

Output:

```text
data/processed/selected_features.csv
metadata/selected_features.txt
```

---

## 12. Clustering

Run:

```bash
python scripts/07_cluster.py
```

Two approaches are supported:

### K-means

```text
K = 5
```

### Hierarchical clustering

Use the distance/linkage specified by the paper.

Output:

```text
data/processed/cluster_assignments_kmeans.csv
data/processed/cluster_assignments_hierarchical.csv
```

Each row should have:

```text
latitude
longitude
cluster
```

---

## 13. Cluster-wise AdaBoost

The important idea is:

```text
Cluster 1 → AdaBoost
Cluster 2 → AdaBoost
Cluster 3 → AdaBoost
Cluster 4 → AdaBoost
Cluster 5 → AdaBoost
```

For each target:

```text
green_wf
blue_wf
total_wf
```

train the corresponding regression models.

Run:

```bash
python scripts/08_train_models.py
```

Models are saved under:

```text
models/
```

---

## 14. Train/test split

Do not randomly split the data without checking spatial leakage.

For geospatial data, nearby locations can be highly similar. A random split can therefore make test performance look better than genuine geographic generalization.

If the paper specifies a particular split, reproduce that split.

Otherwise document your own strategy clearly.

---

## 15. Evaluation

Run:

```bash
python scripts/09_evaluate.py
```

Metrics:

### MAE

```text
MAE = mean(|y - y_hat|)
```

### MSE

```text
MSE = mean((y - y_hat)^2)
```

### MAPE

```text
MAPE = mean(|(y - y_hat) / y|) * 100
```

Handle zero/near-zero targets explicitly for MAPE.

### R²

```text
R² = 1 - SSE/SST
```

Results are saved to:

```text
results/metrics/
results/predictions/
results/figures/
```

---

## 16. Suggested final outputs

Your final project should produce:

```text
results/
├── metrics/
│   ├── kmeans_green_metrics.csv
│   ├── kmeans_blue_metrics.csv
│   ├── kmeans_total_metrics.csv
│   ├── hierarchical_green_metrics.csv
│   ├── hierarchical_blue_metrics.csv
│   └── hierarchical_total_metrics.csv
│
├── predictions/
│   ├── green_predictions.csv
│   ├── blue_predictions.csv
│   └── total_predictions.csv
│
└── figures/
    ├── cluster_map.png
    ├── actual_vs_predicted_green.png
    ├── actual_vs_predicted_blue.png
    └── actual_vs_predicted_total.png
```

---

## 17. Research questions your implementation answers

### RQ1
Can machine learning predict wheat water footprint from environmental variables?

### RQ2
Does clustering environmental conditions before regression improve prediction?

### RQ3
How do K-means and hierarchical clustering compare?

### RQ4
How accurately can the system predict:

- Green WF?
- Blue WF?
- Total WF?

---

## 18. Recommended implementation order

Do not start by downloading 25 GB of climate data.

Follow this order:

```text
1. Download WF
        ↓
2. Download crop calendar
        ↓
3. Extract wheat cells
        ↓
4. Build master spatial grid
        ↓
5. Validate WF + calendar join
        ↓
6. Create wheat location list
        ↓
7. Test climate processing on PR only
        ↓
8. Validate monthly climate output
        ↓
9. Process remaining 6 climate variables
        ↓
10. Add soil
        ↓
11. Build final 102-feature dataset
        ↓
12. Select 20 features
        ↓
13. K-means + hierarchical clustering
        ↓
14. Cluster-wise AdaBoost
        ↓
15. Evaluate
        ↓
16. Generate maps/plots
        ↓
17. Document results
```

---

## 19. Git workflow

Do not commit large datasets.

```bash
git init

git add README.md src scripts config metadata requirements.txt environment.yml .gitignore

git commit -m "Initial wheat water footprint ML pipeline"

git branch -M main

git remote add origin YOUR_GITHUB_REPOSITORY_URL

git push -u origin main
```

Large raw datasets should remain outside Git or use Git LFS only when legally/technically appropriate.

---

## 20. What to put in GitHub

Commit:

```text
Code
README
Configuration
Feature dictionary
Data-source metadata
Small sample data
Notebooks
Model definitions
Evaluation scripts
Documentation
Results that are small enough to share
```

Do not commit:

```text
Large NetCDF files
Raw climate archives
Large WF archives
Private repository credentials
API tokens
Local virtual environments
```

---

## 21. Reproducibility checklist

Before calling the project complete, record:

- WF dataset version
- WF DOI
- Crop-calendar record/version
- Climate dataset/version
- Soil dataset/version
- Download dates
- Spatial resolution
- Aggregation method
- Climate unit conversions
- Missing-value handling
- Crop-calendar rule
- Feature-selection method
- Final 20 features
- Clustering algorithm
- K value
- Distance metric
- Linkage method
- AdaBoost parameters
- Train/test split
- Random seeds
- Evaluation metrics

This information should be stored in `metadata/DATA_SOURCES.csv` and `config/config.yaml`.

