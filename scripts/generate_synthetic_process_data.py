"""Generate synthetic semiconductor process data for Module B.

Output: data/synthetic/process_data.csv  (default)

All generated data is SIMULATED. It does not represent real fab measurements.

Usage:
    python scripts/generate_synthetic_process_data.py
    python scripts/generate_synthetic_process_data.py --n-lots 100 --seed 123
    python scripts/generate_synthetic_process_data.py --output path/to/out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a top-level script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semiconductor_yield.config import SYNTHETIC_DIR, ensure_output_dirs
from semiconductor_yield.process.synthetic import PROCESS_STEPS, SyntheticProcessDataGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic semiconductor process data (Module B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-lots",       type=int,   default=50,    help="Number of lots")
    p.add_argument("--n-wafers",     type=int,   default=25,    help="Wafers per lot")
    p.add_argument("--seed",         type=int,   default=42,    help="Random seed (for reproducibility)")
    p.add_argument("--anomaly-rate", type=float, default=0.05,  help="Target anomaly fraction")
    p.add_argument(
        "--anomaly-types",
        nargs="+",
        default=["drift", "spike", "step_shift", "tool_offset"],
        help="Anomaly types to inject",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/synthetic/process_data.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ensure_output_dirs()
    out_path = Path(args.output) if args.output else SYNTHETIC_DIR / "process_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Synthetic Process Data Generator")
    print("NOTE: All output is SIMULATED — not real fab data.")
    print("=" * 60)
    print(f"  Lots           : {args.n_lots}")
    print(f"  Wafers/lot     : {args.n_wafers}")
    print(f"  Process steps  : {PROCESS_STEPS}")
    print(f"  Anomaly types  : {args.anomaly_types}")
    print(f"  Random seed    : {args.seed}")
    print(f"  Output         : {out_path}")
    print()

    gen = SyntheticProcessDataGenerator(seed=args.seed)
    df = gen.generate(
        n_lots=args.n_lots,
        n_wafers_per_lot=args.n_wafers,
        anomaly_rate=args.anomaly_rate,
        anomaly_types=args.anomaly_types,
    )

    df.to_csv(out_path, index=False, encoding="utf-8")

    # ── Summary ────────────────────────────────────────────────────────────────
    n_anomalies = int(df["anomaly_label"].sum())
    print(f"Saved {len(df):,} rows → {out_path}")
    print()
    print("Dataset summary:")
    print(f"  Total rows        : {len(df):,}")
    print(f"  Unique lots       : {df['lot_id'].nunique()}")
    print(f"  Unique wafers     : {df['wafer_id'].nunique()}")
    print(f"  Anomalous rows    : {n_anomalies:,}  ({n_anomalies / len(df):.1%})")
    print()
    print("Anomaly type breakdown:")
    for atype, count in df["anomaly_type"].value_counts().items():
        print(f"  {atype:<15}: {count:>5,} rows")
    print()
    print("Parameter ranges (non-NaN values):")
    param_cols = ["temperature", "pressure", "gas_flow", "rf_power",
                  "film_thickness", "yield_rate"]
    for col in param_cols:
        vals = df[col].dropna()
        if len(vals):
            print(f"  {col:<20}: [{vals.min():.2f}, {vals.max():.2f}]  mean={vals.mean():.2f}")


if __name__ == "__main__":
    main()
