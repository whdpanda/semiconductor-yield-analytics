"""Run process RCA candidate analysis and generate a Markdown report.

Usage:
    python scripts/run_process_rca.py
    python scripts/run_process_rca.py --feature-set process_only
    python scripts/run_process_rca.py --data-source secom

Reads:
  data/synthetic/process_data.csv           (or --data)
  outputs/reports/process/spc_violations.csv (optional — from run_spc_analysis.py)
  outputs/reports/process/anomaly_scores.csv (optional — from evaluate_process_anomaly.py)

Writes:
  outputs/reports/process/rca_report.md

The report lists root cause CANDIDATES only — it does NOT confirm a root cause.
Real fab diagnosis requires recipe review, tool logs, and engineer domain review.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger

from semiconductor_yield.config import PROCESS_REPORTS_DIR, SYNTHETIC_DIR
from semiconductor_yield.process.rca import LIMITATION_NOTE, analyze
from semiconductor_yield.process.report import generate_markdown_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate RCA candidate report for process anomaly data"
    )
    parser.add_argument(
        "--data", type=Path, default=SYNTHETIC_DIR / "process_data.csv",
        help="Process data CSV (default: data/synthetic/process_data.csv)",
    )
    parser.add_argument(
        "--spc-violations", type=Path,
        default=PROCESS_REPORTS_DIR / "spc_violations.csv",
        help="SPC violations CSV (optional — skip if not present)",
    )
    parser.add_argument(
        "--anomaly-scores", type=Path,
        default=PROCESS_REPORTS_DIR / "anomaly_scores.csv",
        help="Anomaly scores CSV from evaluate_process_anomaly.py (optional)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROCESS_REPORTS_DIR / "rca_report.md",
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--data-source", choices=["synthetic", "secom"], default="synthetic",
        help="'synthetic' allows named steps; 'secom' uses anonymous feature labels",
    )
    parser.add_argument(
        "--top-n", type=int, default=3,
        help="Maximum number of candidates to report (default 3)",
    )
    parser.add_argument(
        "--feature-set", choices=["process_only", "full"], default="full",
        help="Feature set used (informational, shown in report header)",
    )
    args = parser.parse_args(argv)

    # ── Load process data ──────────────────────────────────────────────────────
    if not args.data.exists():
        print(f"\nProcess data not found: {args.data}")
        print("Run: python scripts/generate_synthetic_process_data.py\n")
        return 1

    logger.info(f"Loading process data from {args.data}")
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df):,} rows, {df['lot_id'].nunique()} lots")

    # ── Load SPC violations (optional) ────────────────────────────────────────
    spc_violations: pd.DataFrame | None = None
    if args.spc_violations.exists():
        logger.info(f"Loading SPC violations from {args.spc_violations}")
        spc_violations = pd.read_csv(args.spc_violations)
        logger.info(f"  {len(spc_violations)} SPC violations loaded")
    else:
        logger.warning(
            f"SPC violations not found at {args.spc_violations}. "
            "Run scripts/run_spc_analysis.py to generate. Proceeding without SPC evidence."
        )

    # ── Load anomaly scores (optional) ────────────────────────────────────────
    anomaly_scores: pd.DataFrame | None = None
    if args.anomaly_scores.exists():
        logger.info(f"Loading anomaly scores from {args.anomaly_scores}")
        anomaly_scores = pd.read_csv(args.anomaly_scores)
        logger.info(f"  {len(anomaly_scores)} rows of anomaly scores loaded")
    else:
        logger.warning(
            f"Anomaly scores not found at {args.anomaly_scores}. "
            "Run scripts/evaluate_process_anomaly.py to generate. Proceeding without ML evidence."
        )

    if spc_violations is None and anomaly_scores is None:
        print(
            "\nNeither SPC violations nor anomaly scores are available.\n"
            "Run the following scripts first:\n"
            "  python scripts/run_spc_analysis.py\n"
            "  python scripts/train_process_anomaly.py\n"
            "  python scripts/evaluate_process_anomaly.py\n"
        )
        return 1

    # ── Run RCA analysis ───────────────────────────────────────────────────────
    logger.info(f"Running RCA analysis (data_source={args.data_source!r}, top_n={args.top_n})")
    candidates = analyze(
        df=df,
        spc_violations=spc_violations,
        anomaly_scores=anomaly_scores,
        data_source=args.data_source,
        top_n=args.top_n,
    )

    # ── Build metadata for report header ──────────────────────────────────────
    meta: dict = {"feature_set": args.feature_set}
    if anomaly_scores is not None and "anomaly_label" in anomaly_scores.columns:
        y = anomaly_scores["anomaly_label"].astype(int)
        meta["n_samples"] = len(y)
        meta["n_anomalies"] = int(y.sum())
        meta["anomaly_rate"] = float(y.mean())
    if "timestamp" in df.columns:
        meta["analysis_period"] = (
            f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
        )

    # ── Generate report ────────────────────────────────────────────────────────
    report_text = generate_markdown_report(
        candidates=candidates,
        data_source=args.data_source,
        output_path=args.output,
        meta=meta,
    )

    # ── Print summary to stdout ────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  Process RCA Candidate Analysis")
    print(f"{'=' * 64}")
    print(f"  Data source : {args.data_source}")
    print(f"  Candidates  : {len(candidates)}")
    print()

    if candidates:
        print(f"  {'Rank':<4} {'Step':<18} {'Confidence':<10} {'Features'}")
        print(f"  {'-'*4} {'-'*18} {'-'*10} {'-'*30}")
        for i, c in enumerate(candidates, 1):
            feats = ", ".join(c.suspicious_features[:3])
            if len(c.suspicious_features) > 3:
                feats += f" (+{len(c.suspicious_features) - 3} more)"
            print(f"  {i:<4} {c.suspected_process_step:<18} {c.confidence_level:<10} {feats}")
        print()
        print(f"  NOTE: All candidates are STATISTICAL SUGGESTIONS only.")
        print(f"        Engineer review and tool log analysis required before any action.")
    else:
        print("  No candidates generated — check that SPC/anomaly inputs contain data.")

    print(f"\n  Report saved: {args.output}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
