"""MODULE 8 - cleaning, standardisation and the 80:20 split.

Paper reference counts: 17,897 initial locations -> 17,552 complete ->
14,042 train / 3,510 test. Your numbers will differ; report both.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import get_logger
from .grids import validate_coordinates
from .io_utils import RowLedger, save_table

log = get_logger(__name__)


def clean(cfg: dict, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    ledger = RowLedger("preprocess")
    ledger.record("initial observations", len(df))

    df = df.drop_duplicates()
    ledger.record("after duplicate removal", len(df))

    df = validate_coordinates(df)
    ledger.record("after coordinate validation", len(df))

    df = df[df[target_col].notna() & np.isfinite(df[target_col]) & (df[target_col] > 0)]
    ledger.record(f"after valid {target_col}", len(df))

    if cfg["preprocess"]["drop_incomplete"]:
        df = df.dropna()
        ledger.record("after incomplete-record removal", len(df),
                      "matches the paper's 17,897 -> 17,552 step")

    ref = cfg["preprocess"]["paper_reference"]
    log.info("Paper reference: %d initial / %d clean. You have %d clean.",
             ref["initial_locations"], ref["after_cleaning"], len(df))

    ledger.write(cfg["paths"]["tables"])
    return df.reset_index(drop=True)


def split_and_scale(cfg: dict, df: pd.DataFrame, target_col: str,
                    feature_cols: list[str] | None = None) -> dict:
    """80:20 split, then fit the scaler on TRAIN ONLY.

    Returns a dict with X_train, X_test (scaled DataFrames), y_train, y_test,
    the fitted scaler, and the original row indices for reproducibility.
    """
    pcfg = cfg["preprocess"]
    targets = list(cfg["target"]["variables"].values())
    feature_cols = feature_cols or [c for c in df.columns if c not in targets]

    X = df[feature_cols]
    y = df[target_col]
    idx = np.arange(len(df))

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, idx, test_size=pcfg["test_size"],
        random_state=cfg["project"]["random_state"], shuffle=True,
    )

    scaler = StandardScaler().fit(X_tr)          # never fit on the test set
    X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=feature_cols, index=X_tr.index)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=feature_cols, index=X_te.index)

    if pcfg["scale_target"]:
        log.warning("scale_target=true - the guide says do NOT scale y unless the "
                    "paper's implementation requires it.")

    if pcfg.get("split_strategy", "random") == "random":
        log.warning("Random split used. README section 14: nearby grid cells are "
                    "spatially correlated, so a random split can flatter test "
                    "performance. Document this choice in docs/methodology.md.")

    ref = pcfg["paper_reference"]
    log.info("Split: train=%d (paper %d), test=%d (paper %d)",
             len(X_tr), ref["train"], len(X_te), ref["test"])

    out_dir = Path(cfg["paths"]["processed"])
    np.save(out_dir / f"train_index_{target_col}.npy", idx_tr)
    np.save(out_dir / f"test_index_{target_col}.npy", idx_te)
    with open(out_dir / f"split_meta_{target_col}.json", "w") as fh:
        json.dump({"target": target_col, "n_train": len(X_tr), "n_test": len(X_te),
                   "test_size": pcfg["test_size"],
                   "random_state": cfg["project"]["random_state"],
                   "n_features": len(feature_cols)}, fh, indent=2)

    return {"X_train": X_tr_s, "X_test": X_te_s, "y_train": y_tr, "y_test": y_te,
            "scaler": scaler, "feature_cols": feature_cols,
            "idx_train": idx_tr, "idx_test": idx_te}
