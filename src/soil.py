"""MODULE 5 - soil attributes from the FAO/UNESCO DSMW.

Source: FAO Soils Portal -> FAO/UNESCO Soil Map of the World ->
        "Digital Soil Map of the World (Geonetwork)".
        A 1:5,000,000 vector product: polygons with a soil-unit code, plus a
        separate attribute key.

IMPORTANT (from the implementation guide): the base paper does not publish a
complete machine-readable list of the soil features it used. Fill
`soil.attributes` in config.yaml ONLY after reading the paper's Table 1 and
supplementary material. Do not invent soil variables.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import get_logger
from .io_utils import save_table

log = get_logger(__name__)


def find_soil_source(raw_dir: str | Path) -> Path:
    """Locate the DSMW shapefile (or a raster fallback such as HWSD)."""
    raw_dir = Path(raw_dir)
    for pattern in ("*.shp", "*.tif", "*.tiff", "*.bil"):
        hits = sorted(raw_dir.rglob(pattern))
        if hits:
            log.info("Soil source: %s", hits[0].name)
            return hits[0]
    raise FileNotFoundError(
        f"No soil shapefile/raster in {raw_dir}. See docs/DOWNLOAD_GUIDE.md section 4."
    )


def sample_soil_polygons(points: pd.DataFrame, shp_path: Path,
                         attributes: list[str]) -> pd.DataFrame:
    """Spatial join: which DSMW polygon contains each wheat grid-cell centre."""
    import geopandas as gpd
    from shapely.geometry import Point

    soil = gpd.read_file(shp_path)
    log.info("DSMW columns: %s", list(soil.columns))
    if not attributes:
        raise ValueError(
            "config soil.attributes is empty. Inspect the columns logged above, "
            "confirm against the paper's Table 1, then list them in config.yaml. "
            "The guide forbids inventing soil variables."
        )
    missing = [a for a in attributes if a not in soil.columns]
    if missing:
        raise KeyError(f"Attributes not in DSMW: {missing}. Available: {list(soil.columns)}")

    pts = gpd.GeoDataFrame(
        points[["lat", "lon"]].copy(),
        geometry=[Point(xy) for xy in zip(points["lon"], points["lat"])],
        crs="EPSG:4326",
    )
    if soil.crs is not None and soil.crs.to_string() != "EPSG:4326":
        soil = soil.to_crs("EPSG:4326")

    joined = gpd.sjoin(pts, soil[attributes + ["geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    out = joined[attributes].reset_index(drop=True)
    n_unmatched = int(out[attributes[0]].isna().sum())
    log.info("Soil join: %d/%d matched (%d unmatched, likely coastal cells)",
             len(out) - n_unmatched, len(out), n_unmatched)
    return out


def encode_categoricals(df: pd.DataFrame, how: str = "onehot") -> pd.DataFrame:
    """Turn soil-class strings into numbers, only if the modelling step needs it."""
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    if not cat_cols or how == "none":
        return df
    if how == "onehot":
        out = pd.get_dummies(df, columns=cat_cols, prefix=[f"soil_{c}" for c in cat_cols],
                             dummy_na=False, dtype=int)
    elif how == "ordinal":
        out = df.copy()
        for c in cat_cols:
            out[c] = out[c].astype("category").cat.codes.replace(-1, np.nan)
    else:
        raise ValueError(f"Unknown encoding: {how}")
    log.info("Encoded %d categorical soil column(s) -> %d columns",
             len(cat_cols), out.shape[1])
    return out


def build_soil_features(cfg: dict, points: pd.DataFrame) -> pd.DataFrame:
    src = find_soil_source(cfg["paths"]["raw_soil"])
    raw = sample_soil_polygons(points, src, cfg["soil"]["attributes"])
    out = encode_categoricals(raw, cfg["soil"]["categorical_encoding"])
    out = out.add_prefix("soil_") if not out.columns.str.startswith("soil_").all() else out
    save_table(out, Path(cfg["paths"]["interim"]) / cfg["output"]["soil_table"])
    return out
