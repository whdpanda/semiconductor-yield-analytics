"""Compare evaluation metrics between two WaferCNN training runs.

Usage:
    python scripts/compare_wafer_runs.py \\
        --baseline-run-id run_20240101_120000 \\
        --candidate-run-id run_20240102_090000

Reads:
    outputs/reports/wafer/runs/<baseline_run_id>/evaluation_metrics.json
    outputs/reports/wafer/runs/<candidate_run_id>/evaluation_metrics.json

Outputs:
    outputs/reports/wafer/run_comparison_<candidate_run_id>.json
    outputs/reports/wafer/run_comparison_<candidate_run_id>.csv

Metrics compared:
    accuracy, macro_f1, weighted_f1, none_recall, false_alarm_rate,
    defect_recall, scratch_precision, scratch_recall

Per-metric verdict:
    improved   -- candidate meaningfully better (Δ ≥ 0.02 for beneficial direction)
    regressed  -- candidate meaningfully worse  (Δ ≥ 0.02)
    similar    -- difference < 0.02

Overall verdict:
    improved   -- primary targets (none_recall ↑, false_alarm_rate ↓) improved,
                  macro_f1 not severely regressed (≥ baseline × 0.75)
    regressed  -- primary targets worsened
    trade-off  -- mixed results (some improved, some regressed)

NOTE: This is a portfolio project using the public WM-811K dataset.
      Comparison metrics do not represent real fab deployment performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semiconductor_yield.config import WAFER_REPORTS_DIR

# Metrics where higher is better
_HIGHER_IS_BETTER = {
    "accuracy", "macro_f1", "weighted_f1",
    "none_recall", "defect_recall",
    "scratch_precision", "scratch_recall",
}
# Metrics where lower is better
_LOWER_IS_BETTER = {"false_alarm_rate"}

_COMPARISON_METRICS = list(_HIGHER_IS_BETTER | _LOWER_IS_BETTER)

# Threshold for "meaningful" improvement / regression
_DELTA_THRESHOLD = 0.02


def _load_metrics(run_id: str, reports_root: Path) -> dict:
    path = reports_root / "runs" / run_id / "evaluation_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"evaluation_metrics.json not found for run '{run_id}'.\n"
            f"Expected: {path}\n"
            "Run 'python scripts/train_wafer_cnn.py' to generate it."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _per_metric_verdict(metric: str, baseline_val: float, candidate_val: float) -> str:
    delta = candidate_val - baseline_val
    if metric in _HIGHER_IS_BETTER:
        if delta >= _DELTA_THRESHOLD:
            return "improved"
        if delta <= -_DELTA_THRESHOLD:
            return "regressed"
        return "similar"
    else:  # lower is better
        if delta <= -_DELTA_THRESHOLD:
            return "improved"
        if delta >= _DELTA_THRESHOLD:
            return "regressed"
        return "similar"


def _overall_verdict(
    baseline: dict,
    candidate: dict,
    per_metric: dict[str, dict],
) -> str:
    none_recall_verdict    = per_metric["none_recall"]["verdict"]
    false_alarm_verdict    = per_metric["false_alarm_rate"]["verdict"]
    macro_f1_verdict       = per_metric["macro_f1"]["verdict"]
    defect_recall_verdict  = per_metric["defect_recall"]["verdict"]

    # Primary targets: none_recall and false_alarm_rate
    primary_ok = (
        none_recall_verdict == "improved"
        and false_alarm_verdict == "improved"
    )
    # Guardrail: macro_f1 must not collapse (candidate ≥ 75% of baseline)
    baseline_f1 = baseline.get("macro_f1", 0.0)
    candidate_f1 = candidate.get("macro_f1", 0.0)
    f1_catastrophic = (
        baseline_f1 > 0.0 and candidate_f1 < baseline_f1 * 0.75
    )

    if primary_ok and not f1_catastrophic:
        return "improved"

    # Count regressions across all metrics
    n_improved  = sum(1 for m in per_metric.values() if m["verdict"] == "improved")
    n_regressed = sum(1 for m in per_metric.values() if m["verdict"] == "regressed")

    if n_regressed > n_improved:
        return "regressed"
    if n_improved > 0 and n_regressed > 0:
        return "trade-off"
    if n_regressed == 0:
        return "improved"
    return "regressed"


def compare(
    baseline_run_id: str,
    candidate_run_id: str,
    reports_root: Path | None = None,
) -> dict:
    """Load and compare two runs; return comparison dict."""
    if reports_root is None:
        reports_root = Path(WAFER_REPORTS_DIR)

    baseline  = _load_metrics(baseline_run_id,  reports_root)
    candidate = _load_metrics(candidate_run_id, reports_root)

    per_metric: dict[str, dict] = {}
    for metric in sorted(_COMPARISON_METRICS):
        b_val = float(baseline.get(metric,  0.0))
        c_val = float(candidate.get(metric, 0.0))
        delta = round(c_val - b_val, 4)
        verdict = _per_metric_verdict(metric, b_val, c_val)
        per_metric[metric] = {
            "baseline":  round(b_val, 4),
            "candidate": round(c_val, 4),
            "delta":     delta,
            "verdict":   verdict,
        }

    overall = _overall_verdict(baseline, candidate, per_metric)

    result = {
        "disclaimer": (
            "Comparison on WM-811K public dataset (portfolio project — "
            "not real fab deployment performance)."
        ),
        "baseline_run_id":  baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "baseline_split":   baseline.get("split",  "val"),
        "candidate_split":  candidate.get("split",  "val"),
        "overall_verdict":  overall,
        "delta_threshold":  _DELTA_THRESHOLD,
        "metrics":          per_metric,
    }
    return result


def save_comparison(result: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write comparison JSON and CSV to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cand_id = result["candidate_run_id"]

    json_path = output_dir / f"run_comparison_{cand_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    csv_path = output_dir / f"run_comparison_{cand_id}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "baseline", "candidate", "delta", "verdict"]
        )
        writer.writeheader()
        for metric, m in result["metrics"].items():
            writer.writerow({"metric": metric, **m})

    return json_path, csv_path


def _print_comparison(result: dict) -> None:
    print(f"\n{'=' * 65}")
    print(f"  Run Comparison")
    print(f"{'=' * 65}")
    print(f"  Baseline  : {result['baseline_run_id']}")
    print(f"  Candidate : {result['candidate_run_id']}")
    print(f"  Overall   : {result['overall_verdict'].upper()}")
    print(f"\n  {'Metric':<22}  {'Baseline':>9}  {'Candidate':>9}  {'Δ':>7}  Verdict")
    print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*9}")
    for metric, m in result["metrics"].items():
        flag = {"improved": "↑", "regressed": "✗", "similar": "·"}.get(m["verdict"], "?")
        print(
            f"  {metric:<22}  {m['baseline']:>9.4f}  {m['candidate']:>9.4f}"
            f"  {m['delta']:>+7.4f}  {flag} {m['verdict']}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics between two WaferCNN runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-run-id",  required=True, help="Baseline run_id string")
    parser.add_argument("--candidate-run-id", required=True, help="Candidate run_id string")
    parser.add_argument(
        "--reports-dir", type=Path, default=None,
        help="Root reports directory (default: outputs/reports/wafer)",
    )
    args = parser.parse_args(argv)

    reports_root = args.reports_dir or Path(WAFER_REPORTS_DIR)

    try:
        result = compare(args.baseline_run_id, args.candidate_run_id, reports_root)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    _print_comparison(result)

    output_dir = reports_root
    json_path, csv_path = save_comparison(result, output_dir)
    print(f"  Comparison JSON : {json_path}")
    print(f"  Comparison CSV  : {csv_path}")
    print()
    print("NOTE: This is a portfolio project using the public WM-811K dataset.")
    print("      Comparison metrics do not represent real fab deployment performance.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
