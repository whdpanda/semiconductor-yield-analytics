"""Run SPC analysis on synthetic process data.

Usage:
    python scripts/run_spc_analysis.py
    python scripts/run_spc_analysis.py --output-dir outputs/reports/process

If data/synthetic/process_data.csv does not exist, run:
    python scripts/generate_synthetic_process_data.py
"""

import matplotlib
matplotlib.use("Agg")  # headless — must be set before any other matplotlib import

import argparse
import sys
from pathlib import Path

# Allow running as a top-level script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger

from semiconductor_yield.config import PROCESS_REPORTS_DIR, SYNTHETIC_DIR
from semiconductor_yield.process.spc import WE_RULES, run_spc
from semiconductor_yield.process.visualization import plot_spc_summary

# Features to monitor — same list the synthetic generator writes
MONITOR_COLS: list[str] = [
    "temperature", "pressure", "gas_flow", "rf_power",
    "exposure_dose", "film_thickness", "overlay_error", "defect_density",
]

_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _print_summary(violations_df: pd.DataFrame) -> None:
    n = len(violations_df)
    print(f"\n{'=' * 60}")
    print(f"  SPC Analysis Summary")
    print(f"{'=' * 60}")
    print(f"  Total SPC signals : {n}")

    if n == 0:
        print("  No violations detected — process in control.")
        return

    print(f"\n  By rule:")
    for rule in WE_RULES:
        cnt = int((violations_df["rule"] == rule).sum())
        bar = "█" * min(cnt // 5, 40)
        print(f"    {rule:<8}  {cnt:5d}  {bar}")

    print(f"\n  By severity:")
    for sev in ["HIGH", "MEDIUM", "LOW"]:
        cnt = int((violations_df["severity"] == sev).sum())
        print(f"    {sev:<8}  {cnt:5d}")

    print(f"\n  By feature (top 5):")
    top = violations_df["feature"].value_counts().head(5)
    for feat, cnt in top.items():
        print(f"    {feat:<20}  {cnt:5d}")

    print(f"\n  By process step:")
    for step, cnt in violations_df["process_step"].value_counts().items():
        print(f"    {step:<20}  {cnt:5d}")

    # Show highest-severity violations
    high = violations_df[violations_df["severity"] == "HIGH"]
    if len(high) > 0:
        print(f"\n  HIGH severity signals (Rule 1 — beyond 3σ):")
        shown = high[["timestamp", "process_step", "feature", "value"]].head(10)
        print(shown.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SPC analysis on synthetic process data")
    parser.add_argument(
        "--data",
        type=Path,
        default=SYNTHETIC_DIR / "process_data.csv",
        help="Path to process_data.csv (default: data/synthetic/process_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESS_REPORTS_DIR,
        help="Directory for SPC report CSV and control chart PNGs",
    )
    args = parser.parse_args(argv)

    # ── Check data file ────────────────────────────────────────────────────────
    if not args.data.exists():
        print(
            f"\nProcess data not found: {args.data}\n\n"
            "Generate it first:\n"
            "  python scripts/generate_synthetic_process_data.py\n"
        )
        return 1

    # ── Load ───────────────────────────────────────────────────────────────────
    logger.info(f"Loading process data from {args.data}")
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df):,} rows | steps: {sorted(df['process_step'].unique())}")

    feature_cols = [c for c in MONITOR_COLS if c in df.columns]

    # ── Run SPC ────────────────────────────────────────────────────────────────
    violations_df, limits_map = run_spc(df, feature_cols=feature_cols)

    # ── Save report ────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "spc_violations.csv"
    violations_df.to_csv(report_path, index=False)
    logger.info(f"Violations report saved → {report_path}")

    # ── Generate charts ────────────────────────────────────────────────────────
    charts_dir = args.output_dir / "charts"
    created = plot_spc_summary(
        df=df,
        limits_map=limits_map,
        violations_df=violations_df,
        output_dir=charts_dir,
    )
    logger.info(f"Control charts saved → {charts_dir}  ({len(created)} files)")

    # ── Print summary ──────────────────────────────────────────────────────────
    _print_summary(violations_df)
    print(f"\n  Report  : {report_path}")
    print(f"  Charts  : {charts_dir}  ({len(created)} PNGs)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
