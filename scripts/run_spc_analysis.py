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
import json
import sys
from datetime import datetime
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


def _print_summary(violations_df: pd.DataFrame, events_df: pd.DataFrame, n_rows: int) -> None:
    n = len(violations_df)
    n_events = len(events_df)
    print(f"\n{'=' * 60}")
    print(f"  SPC Analysis Summary")
    print(f"{'=' * 60}")
    print(f"  Total SPC signals (raw)   : {n}")
    print(f"  Distinct SPC events       : {n_events}  (Rule 4 deduplicated)")
    print(f"  Violation rate            : {n / n_rows:.1%}  ({n}/{n_rows} rows)")

    if n == 0:
        print("  No violations detected — process in control.")
        return

    print(f"\n  By rule (raw counts):")
    for rule in WE_RULES:
        cnt = int((violations_df["rule"] == rule).sum())
        evt = int((events_df["rule"] == rule).sum())
        bar = "█" * min(cnt // 5, 40)
        if rule == "Rule 4":
            print(f"    {rule:<8}  {cnt:5d}  (events: {evt})  {bar}")
        else:
            print(f"    {rule:<8}  {cnt:5d}  {bar}")

    print(f"\n  Rule 4 note: {cnt // max(evt, 1):.0f}× raw-to-event ratio indicates sustained drift.")
    print(f"  Rule 4 fires at every qualifying 8-point window — many signals = one long run.")

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

    # ── Simulated data notice ──────────────────────────────────────────────────
    print("=" * 60)
    print("  SPC Analysis — Module B")
    print("  NOTE: All process data is SIMULATED.")
    print("  This is a portfolio project, not real fab data.")
    print("=" * 60)
    print()

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

    # ── Run SPC (raw — every sliding-window violation, used for charts) ────────
    violations_df, limits_map = run_spc(df, feature_cols=feature_cols)

    # ── Run SPC (deduplicated Rule 4 — one event per qualifying run) ───────────
    events_df, _ = run_spc(df, feature_cols=feature_cols, deduplicate_rule4=True)

    # ── Save violations CSV ────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "spc_violations.csv"
    violations_df.to_csv(report_path, index=False, encoding="utf-8")
    logger.info(f"Violations report saved → {report_path}")

    # ── Save events CSV (Rule 4 deduplicated) ──────────────────────────────────
    events_path = args.output_dir / "spc_events.csv"
    events_df.to_csv(events_path, index=False, encoding="utf-8")
    logger.info(f"Events report (Rule 4 deduplicated) saved → {events_path}")

    # ── Compute top violating (step, feature) groups ───────────────────────────
    n_violations = len(violations_df)
    n_events = len(events_df)

    if n_violations > 0:
        top_groups = (
            violations_df.groupby(["process_step", "feature"])
            .size()
            .nlargest(10)
            .reset_index(name="violation_count")
        )
        top_violating_groups = [
            {
                "group": f"{row['process_step']}__{row['feature']}",
                "process_step": row["process_step"],
                "feature": row["feature"],
                "violation_count": int(row["violation_count"]),
                "event_count": int(
                    len(events_df[
                        (events_df["process_step"] == row["process_step"])
                        & (events_df["feature"] == row["feature"])
                    ])
                ),
            }
            for _, row in top_groups.iterrows()
        ]
    else:
        top_violating_groups = []

    # ── Save summary JSON ──────────────────────────────────────────────────────
    rule_counts = violations_df["rule"].value_counts().to_dict() if n_violations else {}
    event_rule_counts = events_df["rule"].value_counts().to_dict() if n_events else {}

    summary: dict = {
        "disclaimer": (
            "Analysis of SIMULATED process data. "
            "SPC signals are process control warnings, not confirmed root causes. "
            "This is a portfolio project — not real fab production data."
        ),
        "data_source":          str(args.data),
        "analysis_timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows":               len(df),
        "n_steps":              int(df["process_step"].nunique()),
        "n_features_monitored": len(feature_cols),
        "n_groups_monitored":   len(limits_map),
        # ── Violation counts ───────────────────────────────────────────────────
        "raw_violation_count":  n_violations,
        "event_count":          n_events,
        "total_violations":     n_violations,   # alias kept for dashboard compatibility
        "violation_rate":       round(n_violations / max(len(df), 1), 4),
        # ── By-rule breakdown ──────────────────────────────────────────────────
        "violations_by_rule":      rule_counts,
        "events_by_rule":          event_rule_counts,
        "violations_by_rule_rate": {
            rule: round(cnt / max(len(df), 1), 4)
            for rule, cnt in rule_counts.items()
        },
        # ── Rule 4 interpretation note ─────────────────────────────────────────
        "note_for_rule4": (
            "Rule 4 fires at every index where the 8-point sliding window "
            "contains 8 consecutive same-side points. A sustained drift of N "
            "same-side points produces N-7 raw violations but only 1 event "
            "(see event_count / spc_events.csv). "
            "The high Rule 4 count in this dataset is expected: the synthetic "
            "generator intentionally injects a temperature drift of up to +6 °C "
            "over the last 20 %% of lots. For Lithography (std=0.30 °C) and "
            "Metrology (std=0.50 °C), this equates to a +20σ / +12σ sustained "
            "shift — exactly the pattern Rule 4 is designed to detect. "
            "Rule 4 severity is LOW; these signals direct investigation, they "
            "do not confirm a defect."
        ),
        # ── By-severity / step / feature ──────────────────────────────────────
        "violations_by_severity": (
            violations_df["severity"].value_counts().to_dict() if n_violations else {}
        ),
        "violations_by_step": (
            violations_df["process_step"].value_counts().to_dict() if n_violations else {}
        ),
        "violations_by_feature": (
            violations_df["feature"].value_counts().to_dict() if n_violations else {}
        ),
        # ── Top groups ─────────────────────────────────────────────────────────
        "top_violating_groups": top_violating_groups,
        # ── Per-group control limits ───────────────────────────────────────────
        "control_limits": {
            f"{step}__{feature}": {
                "mean":           round(lim.mean, 6),
                "ucl":            round(lim.ucl, 6),
                "lcl":            round(lim.lcl, 6),
                "non_null_count": int(
                    (df[df["process_step"] == step][feature]
                     .notna()
                     .sum())
                ),
                "n_violations":   int(
                    len(violations_df[
                        (violations_df["feature"] == feature)
                        & (violations_df["process_step"] == step)
                    ])
                ),
                "n_events":       int(
                    len(events_df[
                        (events_df["feature"] == feature)
                        & (events_df["process_step"] == step)
                    ])
                ),
            }
            for (feature, step), lim in sorted(limits_map.items())
        },
    }
    summary_path = args.output_dir / "spc_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved → {summary_path}")

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
    _print_summary(violations_df, events_df, n_rows=len(df))
    print(f"\n  Violations CSV : {report_path}")
    print(f"  Events CSV     : {events_path}")
    print(f"  Summary JSON   : {summary_path}")
    print(f"  Charts         : {charts_dir}  ({len(created)} PNGs)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
