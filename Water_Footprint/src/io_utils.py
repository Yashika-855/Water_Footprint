"""Shared IO helpers: tabular round-trips and a row-count audit trail.

The guide requires you to record how many rows survive every filtering and
joining step. `RowLedger` does that automatically so you cannot forget.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import get_logger

log = get_logger(__name__)


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write .parquet or .csv depending on the suffix; create parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=index)
    elif path.suffix == ".csv":
        df.to_csv(path, index=index)
    else:
        raise ValueError(f"Unsupported table suffix: {path.suffix}")
    log.info("Saved %s  (%d rows x %d cols)", path.name, len(df), df.shape[1])
    return path


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run the earlier pipeline stage first "
            f"(see `make help`)."
        )
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


class RowLedger:
    """Record row counts before/after each step, then dump to JSON + CSV.

    Usage
    -----
    >>> ledger = RowLedger("integration")
    >>> ledger.record("wf grid loaded", len(df))
    >>> ledger.record("after climate join", len(df), note="unmatched dropped")
    >>> ledger.write(cfg["paths"]["tables"])
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.entries: list[dict] = []

    def record(self, step: str, n_rows: int, note: str = "") -> None:
        prev = self.entries[-1]["n_rows"] if self.entries else None
        delta = None if prev is None else n_rows - prev
        self.entries.append(
            {"stage": self.stage, "step": step, "n_rows": n_rows,
             "delta": delta, "note": note}
        )
        log.info("[%s] %-42s n=%-8d %s", self.stage, step, n_rows,
                 f"({delta:+d})" if delta is not None else "")

    def write(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"row_ledger_{self.stage}.csv"
        pd.DataFrame(self.entries).to_csv(csv_path, index=False)
        with open(out_dir / f"row_ledger_{self.stage}.json", "w") as fh:
            json.dump(self.entries, fh, indent=2)
        return csv_path
