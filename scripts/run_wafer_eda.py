"""Run EDA on the WM-811K wafer map dataset and save reports.

Outputs:
    outputs/reports/wafer/class_distribution.csv
    outputs/reports/wafer/class_distribution.png
    outputs/reports/wafer/imbalance_report.csv
    outputs/reports/wafer/sample_wafer_maps.png
    outputs/reports/wafer/wafer_map_sizes.csv

Usage:
    python scripts/run_wafer_eda.py
    python scripts/run_wafer_eda.py --output-dir custom/path/
    python scripts/run_wafer_eda.py --max-samples 5000   # fast smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a top-level script without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set non-interactive backend before any matplotlib import so the script
# works on headless servers without a DISPLAY environment variable.
import matplotlib
matplotlib.use("Agg")

from semiconductor_yield.config import WAFER_REPORTS_DIR, WM811K_PKL, ensure_output_dirs
from semiconductor_yield.wafer.data_loader import WM811KLoader
from semiconductor_yield.wafer.eda import run_eda
from semiconductor_yield.wafer.preprocess import imbalance_stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EDA for WM-811K wafer map dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=f"Report output directory (default: {WAFER_REPORTS_DIR})",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit labeled samples loaded (e.g. 5000 for a fast smoke run)",
    )
    return p.parse_args()


def _check_data_file() -> None:
    if not WM811K_PKL.exists():
        print(
            "\n[ERROR] WM-811K dataset not found.\n\n"
            f"  Expected: {WM811K_PKL}\n\n"
            "  To download:\n"
            "    1. Go to https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map\n"
            "    2. Download LSWMD.pkl  (~350 MB)\n"
            "    3. Place it at:  data/raw/wm811k/LSWMD.pkl\n",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else WAFER_REPORTS_DIR

    _check_data_file()
    ensure_output_dirs()

    print("=" * 60)
    print("WM-811K Wafer Map EDA")
    print("=" * 60)

    loader = WM811KLoader()

    print("Loading raw data …")
    df_raw = loader.load_raw()
    print(f"  Raw DataFrame: {len(df_raw):,} rows")

    print("Parsing labeled samples …")
    samples = loader.load(labeled_only=True)
    print(f"  Labeled samples: {len(samples):,}")

    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples[: args.max_samples]
        print(f"  Capped at {len(samples):,} (--max-samples)")

    # Print quick summary to stdout
    print("\nClass distribution:")
    stats = imbalance_stats(samples)
    for _, row in stats.iterrows():
        bar = "█" * int(row["percentage"] / 2)
        print(
            f"  {row['class_name']:<12} {row['count']:>7,}  "
            f"({row['percentage']:5.1f}%)  {bar}"
        )

    print(f"\nRunning EDA → {output_dir}\n")
    outputs = run_eda(samples, df_raw=df_raw, output_dir=output_dir)

    print("\nSaved artefacts:")
    for key, path in outputs.items():
        print(f"  {key:<25}  {path.relative_to(Path.cwd()) if path.is_absolute() else path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
