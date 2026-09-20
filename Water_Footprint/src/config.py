"""Configuration loading and path management."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

# Repo root = one level above this file (src/config.py -> repo/)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml and resolve every path entry to an absolute Path."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["paths"] = {k: (ROOT / v) for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    cfg["_root"] = ROOT
    return cfg


def get_logger(name: str) -> logging.Logger:
    """Consistent console logger for every script."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
