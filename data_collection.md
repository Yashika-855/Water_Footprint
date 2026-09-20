# Data Collection

Where each dataset lives, exactly what to download, and in what order.

Four datasets, in the order you should actually collect them. Steps 1 and 2 are
small and quick; do them first and get the pipeline working before you touch the
climate archive.

Total disk: budget **~50 GB** free.

---

## 1. Water footprint (target variable) — ~150 MB

**Where:** 4TU.ResearchData, DOI `10.4121/7b45bcc6-686b-404d-a910-13c87156716a`
<https://data.4tu.nl/datasets/7b45bcc6-686b-404d-a910-13c87156716a>

**What it is:** outputs of the ACEA process-based gridded crop model
(Mialyk et al.), 175 crops, 1990–2019, 5 arcminute resolution, water footprints
partitioned into green and blue, unit WF expressed in m³/t.

**Download:**

| File | Size | Why |
|---|---|---|
| `readme.pdf` | 0.6 MB | **Read first.** Confirms variable names and whether wheat is in the "selected crops" subset. |
| `unit_wf_selected_crops_average_2010_2019.zip` | ~148 MB | Unit WF in m³/t, already 2010–2019 averaged. This is your target. |

Fallback if wheat is absent from the selected-crops subset:
`wf_crop_production_1990_2019.zip` (~359 MB) — but then you must derive unit WF
and average the years yourself.

**Do not** use `national_wf_175_crops_*.csv`. The guide forbids substituting
national averages for the gridded target.

**Version matters.** The dataset has v1 (2023), v2 (2024) and v3 (2025).
Record which one you took in `metadata/DATA_SOURCES.csv`.

Unzip into `data/raw/wf/`, keep only wheat files, never edit them.

---

## 2. Crop calendar — ~3 MB (automated)

**Where:** Zenodo record 5062513, GGCMI Phase 3 crop calendar
(Jägermeyr et al. 2021).

`python scripts/00_download_data.py` fetches this for you via the Zenodo API.

**What it is:** planting day and maturity day in each 0.5° land grid cell for
18 crops, separating rainfed and irrigated systems, with separate calendars for
winter and spring wheat.

Files you need (codes as used by ISIMIP's mirror of the same data):

- `swh` = spring wheat, `wwh` = winter wheat
- `noirr` / `rf` = rainfed, `firr` / `ir` = irrigated

Four files total. Both variables are day-of-year integers — that is what feeds
the seven-day cultivation rule.

---

## 3. Climate — 15–25 GB (the hard one)

**Where:** ISIMIP Repository <https://data.isimip.org> — free account required.

Path: `ISIMIP3a / InputData / climate / atmosphere / obsclim / global / daily /
historical / GSWP3-W5E5`

**What it is:** daily data on a 0.5° lat-lon grid; obsclim covers 1901–2019 and
includes `hurs`, `sfcWind`, `tasmax`, `tasmin`, `rsds`, `ps` and `pr` — exactly
the seven variables the paper uses.

### Do this, in this order

1. **Build the wheat location list first** (`python scripts/01_prepare_wf.py`). The global WF grid
   has millions of cells; wheat occupies a small fraction, and the paper ends up
   with ~17,900 locations. Once you have that list you only ever need climate at
   those points.
2. **Use the repository's file cutout service** to request a bounding box rather
   than the whole globe.
3. **Download `pr` only.** Run `python scripts/03_climate.py --variable pr` and
   confirm the output dimensions and units look right.
4. Only then fetch the other six.

Files are chunked by decade, named like
`gswp3-w5e5_obsclim_pr_global_daily_2011_2020.nc`. For 2010–2019 you need the
`2001_2010` **and** `2011_2020` chunks of each variable.

### Units — get these right

| Variable | Native unit | Conversion |
|---|---|---|
| `pr` | kg m⁻² s⁻¹ | × 86400 → mm/day |
| `tasmin`, `tasmax` | K | − 273.15 → °C |
| `ps` | Pa | none |
| `rsds` | W m⁻² | none |
| `hurs` | % | none |
| `sfcWind` | m s⁻¹ | none |

The conversions are wired into `config.yaml` under `climate.variables`.

**Speed tip:** `cdo monsum` / `cdo monmean` on the command line will chew
through these far faster than xarray in a notebook. Both are in `environment.yml`.

---

## 4. Soil — a few hundred MB

**Where:** FAO Soils Portal → FAO/UNESCO Soil Map of the World →
"Digital Soil Map of the World (Geonetwork)".
<https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/faounesco-soil-map-of-the-world/en/>

**What it is:** a 1:5,000,000 vector product — polygons carrying a soil-unit
code, plus a separate key mapping codes to attributes.

### Two warnings

1. The base paper does **not** publish a machine-readable list of the soil
   features it used. Read the paper's Table 1 and supplementary material, then
   fill `soil.attributes` in `config/config.yaml`. `scripts/04_prepare_soil.py` will
   deliberately fail with the available column list printed if you leave it
   empty — that is by design, so you choose rather than invent.
2. DSMW is thin on the hydraulic properties the paper mentions (conductivity,
   porosity). Those usually come from the soil *class* via a lookup table, or
   from **HWSD v1.2** on the same portal — a 30 arc-second raster linked to
   harmonized soil property data including water storage capacity, soil depth,
   textural class and granulometry. If you use HWSD instead, log it as a
   deviation in `docs/methodology.md`.

Keep the `.shp` together with its `.dbf`, `.shx` and `.prj` siblings.

---

## Citation obligations

Every one of these sources requires citation, and the ISIMIP terms of use in
particular ask for explicit crediting of data providers. Keep
`metadata/DATA_SOURCES.csv` current **as you download**, not reconstructed at the end.
