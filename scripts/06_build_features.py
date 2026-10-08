#!/usr/bin/env python

from pathlib import Path

import _bootstrap  # noqa: F401

from src.config import load_config, get_logger
from src.features import build_feature_table
from src.io_utils import load_table

log = get_logger("06_build_features")

if __name__ == "__main__":

    cfg = load_config()

    input_path = Path(
        cfg["paths"]["processed"]
    ) / "master_wheat_dataset_aligned.csv"

    log.info("Loading aligned master dataset: %s", input_path)

    df = load_table(input_path)

    log.info(
        "Loaded %d rows x %d columns",
        len(df),
        len(df.columns)
    )

    features, dictionary = build_feature_table(
        cfg,
        df,
        apply_season_mask=False
    )

    log.info(
        "Feature table created: %d rows x %d columns",
        len(features),
        len(features.columns)
    )

    log.info("Feature engineering completed.")