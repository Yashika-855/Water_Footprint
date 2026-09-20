#!/usr/bin/env python
"""Run the whole pipeline end to end, stopping at the first failure."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("01_prepare_wf.py", []),
    ("02_prepare_calendar.py", []),
    ("03_prepare_climate.py", []),
    ("04_prepare_soil.py", []),
    ("05_build_master_dataset.py", []),
    ("06_select_features.py", []),
    ("07_cluster.py", ["--target", "total_wf"]),
    ("08_train_models.py", ["--target", "total_wf"]),
    ("09_evaluate.py", []),
]

if __name__ == "__main__":
    for script, extra in STAGES:
        print(f"\n{'=' * 72}\n>>> {script} {' '.join(extra)}\n{'=' * 72}")
        rc = subprocess.call([sys.executable, str(HERE / script), *extra])
        if rc != 0:
            print(f"\nFAILED at {script} (exit {rc}). Fix this stage first.")
            sys.exit(rc)
    print("\nPipeline complete. See results/metrics/model_comparison.csv")
