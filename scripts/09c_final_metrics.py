#!/usr/bin/env python
"""Module 13 - evaluate cluster-wise AdaBoost predictions.

Run from the project root (virtual environment active):

    python scripts/09_evaluate.py --inspect     # just show columns/rows of each file
    python scripts/09_evaluate.py               # compute metrics + figures

Reads : results/predictions/{target}_wf_predictions_{method}.csv
Writes: results/metrics/{method}_{target}_metrics.csv
        results/metrics/final_results.csv
        results/metrics/paper_vs_student.csv
        results/figures/{method}_{target}_*.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import final_metrics as ev  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--pred-dir", type=Path, default=PROJECT_ROOT / "results" / "predictions")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--min-true", type=float, default=1e-6,
        help="exclude rows with |actual| <= this value from MAPE only (default 1e-6 = guide definition)",
    )
    parser.add_argument("--inspect", action="store_true", help="only print file shapes and column names")
    return parser.parse_args()


def inspect_files(files: list[Path]) -> None:
    for path in files:
        df = pd.read_csv(path, encoding="utf-8-sig")
        print(f"\n{path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
        print("  columns:", list(df.columns))
        print(df.head(3).to_string(index=False))


def main() -> int:
    args = parse_args()
    files = sorted(args.pred_dir.glob("*_wf_predictions_*.csv"))
    if not files:
        logger.error("No prediction files found in %s", args.pred_dir)
        return 1

    if args.inspect:
        inspect_files(files)
        return 0

    metrics_dir = args.out_dir / "metrics"
    figures_dir = args.out_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict] = []
    done: set[tuple[str, str]] = set()

    for path in files:
        try:
            df = ev.load_predictions(path, split=args.split)
        except (ValueError, FileNotFoundError) as exc:
            logger.error("Skipping %s: %s", path.name, exc)
            continue

        target, method = df["target"].iloc[0], df["method"].iloc[0]
        logger.info("Evaluating %s | %s | n=%d", target, method, len(df))

        table = ev.evaluate_method(df, args.min_true)
        table.insert(0, "method", method)
        table.insert(1, "target", target)
        table.to_csv(metrics_dir / f"{method}_{target}_metrics.csv", index=False)
        overall_rows.append(table[table["scope"] == "overall"].iloc[0].to_dict())
        done.add((target, method))

        stem, title = f"{method}_{target}", f"{method} | {target} WF"
        ev.plot_actual_vs_predicted(df, title, figures_dir / f"{stem}_actual_vs_predicted.png")
        ev.plot_residuals(df, title, figures_dir / f"{stem}_residuals.png")
        if not ev.plot_spatial_error(df, title, figures_dir / f"{stem}_spatial_error.png"):
            logger.info("%s: no latitude/longitude columns, spatial error map skipped", path.name)

    if not overall_rows:
        logger.error("Nothing was evaluated. Run with --inspect to see the file columns.")
        return 1

    final = pd.DataFrame(overall_rows).drop(columns=["scope", "cluster"])
    final.to_csv(metrics_dir / "final_results.csv", index=False)
    ev.compare_with_paper(final).to_csv(metrics_dir / "paper_vs_student.csv", index=False)

    pd.options.display.float_format = "{:,.3f}".format
    print("\n=== FINAL RESULTS (test set) ===")
    print(final.to_string(index=False))

    missing = [combo for combo in ev.EXPECTED_COMBINATIONS if combo not in done]
    if missing:
        print("\nMissing combinations (no predictions evaluated):")
        for target, method in missing:
            print(f"  - {target} WF / {method}")
    logger.info("Saved results to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
