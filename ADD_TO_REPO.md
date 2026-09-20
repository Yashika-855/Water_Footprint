# How to add these files to Yashika-855/Water_Footprint

These files are laid out to match the structure your existing `README.md`
already documents (flat `src/`, `metadata/`, `scripts/01–09`, `results/metrics/`).

**Not included on purpose** — your repo already has them, and overwriting would
throw away your work:

- `README.md`
- `LICENSE`
- `.gitignore`
- `requirements.txt`
- `environment.yml`

## Steps

```bash
git clone https://github.com/Yashika-855/Water_Footprint.git
cd Water_Footprint

# copy the contents of this folder in (macOS/Linux)
cp -r /path/to/Water_Footprint_files/. .

# Windows PowerShell
# Copy-Item -Path "C:\path\to\Water_Footprint_files\*" -Destination . -Recurse -Force

git status          # review what is about to be added
git add .
git commit -m "Add pipeline: src modules, scripts 01-09, config, metadata, docs, tests"
git push origin main
```

## Check your .gitignore covers these

Before committing, make sure your existing `.gitignore` includes at least:

```gitignore
data/raw/**
data/interim/**
data/processed/**
!**/.gitkeep
models/*.joblib
results/predictions/*.csv
results/figures/*.png
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.venv/
*.nc
*.nc4
*.zip
*.shp
*.dbf
*.shx
```

The `.gitkeep` files keep the empty `data/` and `results/` folders in Git while
the data itself stays out.

## Two things to fix before your first real run

1. **`src/wf.py` → `WF_VARIABLE_CANDIDATES`.** The ACEA NetCDF variable names are
   guessed. Open the provider `readme.pdf`, check the real names, and correct the
   dict.
2. **`config/config.yaml` → `soil.attributes`.** Empty by design.
   `scripts/04_prepare_soil.py` will fail and print the available DSMW columns —
   pick from the paper's Table 1, don't invent.

## Verify it works

```bash
pip install -r requirements.txt
pytest -q                                  # 10 tests, no data needed
python scripts/00_download_data.py         # fetches the 3 MB crop calendar
python scripts/00_data_inventory.py        # after you add the other datasets
```
