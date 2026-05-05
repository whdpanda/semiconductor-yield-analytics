"""Tests for dashboard data-preparation helpers (data_helpers.py).

No Streamlit runtime required — all imports are from data_helpers.py,
which is intentionally free of Streamlit.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semiconductor_yield.dashboard.data_helpers import (
    format_rca_candidates,
    get_anomaly_rate_by_step,
    get_available_charts,
    get_violation_summary,
    load_anomaly_scores,
    load_anomaly_summary,
    load_process_data,
    load_spc_violations,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_violations(n: int = 6) -> pd.DataFrame:
    """Small synthetic SPC violations DataFrame."""
    return pd.DataFrame({
        "timestamp":    pd.date_range("2024-01-01", periods=n, freq="h"),
        "feature":      ["temperature", "pressure", "temperature",
                         "gas_flow",    "pressure",  "rf_power"][:n],
        "process_step": ["Etching",     "Etching",   "Deposition",
                         "Etching",     "CMP",       "Deposition"][:n],
        "rule":         ["Rule 1", "Rule 2", "Rule 1",
                         "Rule 4", "Rule 2", "Rule 1"][:n],
        "severity":     ["HIGH", "MEDIUM", "HIGH",
                         "LOW",  "MEDIUM", "HIGH"][:n],
        "value":        [101.0, 99.5, 105.0, 98.0, 100.5, 102.0][:n],
    })


def _make_scores(n: int = 10) -> pd.DataFrame:
    """Small synthetic anomaly_scores DataFrame."""
    rng = np.random.default_rng(0)
    steps = (["Etching"] * 4 + ["Deposition"] * 3 + ["CMP"] * 3)[:n]
    return pd.DataFrame({
        "lot_id":       [f"LOT_{i:03d}" for i in range(n)],
        "wafer_id":     [f"W_{i}" for i in range(n)],
        "process_step": steps,
        "if_score":     rng.uniform(-0.05, 0.05, n),
        "ae_score":     rng.uniform(0.001, 0.02, n),
        "if_pred":      rng.integers(0, 2, n),
        "ae_pred":      rng.integers(0, 2, n),
        "ensemble_any": rng.integers(0, 2, n),
        "ensemble_all": rng.integers(0, 2, n),
    })


class _FakeCandidate:
    """Minimal stand-in for RCACandidate."""
    def __init__(self, step, confidence, features, evidence, checks=""):
        self.suspected_process_step = step
        self.confidence_level       = confidence
        self.suspicious_features    = features
        self.evidence               = evidence
        self.recommended_checks     = [checks] if checks else []
        self.limitation_note        = "Synthetic data only."


# ══════════════════════════════════════════════════════════════════════════════
# TestLoadFunctions
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadFunctions:
    def test_load_spc_violations_missing_returns_none(self, tmp_path):
        result = load_spc_violations(tmp_path / "nope.csv")
        assert result is None

    def test_load_spc_violations_reads_csv(self, tmp_path):
        df = _make_violations()
        p = tmp_path / "spc.csv"
        df.to_csv(p, index=False)
        result = load_spc_violations(p)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_load_spc_violations_parses_timestamp(self, tmp_path):
        df = _make_violations()
        p = tmp_path / "spc.csv"
        df.to_csv(p, index=False)
        result = load_spc_violations(p)
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

    def test_load_anomaly_scores_missing_returns_none(self, tmp_path):
        assert load_anomaly_scores(tmp_path / "nope.csv") is None

    def test_load_anomaly_scores_reads_csv(self, tmp_path):
        df = _make_scores()
        p = tmp_path / "scores.csv"
        df.to_csv(p, index=False)
        result = load_anomaly_scores(p)
        assert isinstance(result, pd.DataFrame)
        assert "if_score" in result.columns

    def test_load_anomaly_summary_missing_returns_none(self, tmp_path):
        assert load_anomaly_summary(tmp_path / "nope.json") is None

    def test_load_anomaly_summary_reads_json(self, tmp_path):
        data = {"models": {"isolation_forest": {"f1": 0.25}}}
        p = tmp_path / "summary.json"
        with open(p, "w") as f:
            json.dump(data, f)
        result = load_anomaly_summary(p)
        assert result["models"]["isolation_forest"]["f1"] == 0.25

    def test_load_process_data_missing_returns_none(self, tmp_path):
        assert load_process_data(tmp_path / "nope.csv") is None

    def test_load_process_data_reads_csv(self, tmp_path):
        df = pd.DataFrame({"lot_id": ["L1", "L2"], "process_step": ["Etching", "CMP"]})
        p = tmp_path / "proc.csv"
        df.to_csv(p, index=False)
        result = load_process_data(p)
        assert list(result.columns) == ["lot_id", "process_step"]


# ══════════════════════════════════════════════════════════════════════════════
# TestGetViolationSummary
# ══════════════════════════════════════════════════════════════════════════════

class TestGetViolationSummary:
    def test_none_returns_empty(self):
        result = get_violation_summary(None)
        assert result.empty
        assert list(result.columns) == ["process_step", "rule", "count", "features"]

    def test_empty_df_returns_empty(self):
        result = get_violation_summary(pd.DataFrame())
        assert result.empty

    def test_missing_columns_returns_empty(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert get_violation_summary(df).empty

    def test_correct_columns(self):
        result = get_violation_summary(_make_violations())
        assert set(result.columns) == {"process_step", "rule", "count", "features"}

    def test_groups_correctly(self):
        viol = _make_violations()
        result = get_violation_summary(viol)
        # Etching Rule 1 should have count=1 (only one temperature/Etching/Rule1 row)
        # check total counts sum to input rows
        assert result["count"].sum() == len(viol)

    def test_count_is_positive(self):
        result = get_violation_summary(_make_violations())
        assert (result["count"] > 0).all()

    def test_sorted_by_step_then_count(self):
        result = get_violation_summary(_make_violations())
        # For same process_step, higher count comes first
        for step in result["process_step"].unique():
            sub = result[result["process_step"] == step]["count"].tolist()
            assert sub == sorted(sub, reverse=True), \
                f"Counts not sorted descending for step={step}"

    def test_features_are_comma_joined_uniques(self):
        """'features' cell should contain comma-separated unique feature names."""
        viol = _make_violations()
        result = get_violation_summary(viol)
        for _, row in result.iterrows():
            # Each feature name should appear in the features string
            for feat in row["features"].split(", "):
                assert feat in viol["feature"].unique(), \
                    f"Unknown feature '{feat}' in summary"


# ══════════════════════════════════════════════════════════════════════════════
# TestGetAnomalyRateByStep
# ══════════════════════════════════════════════════════════════════════════════

class TestGetAnomalyRateByStep:
    def test_none_returns_empty(self):
        result = get_anomaly_rate_by_step(None)
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = get_anomaly_rate_by_step(pd.DataFrame())
        assert result.empty

    def test_no_step_column_returns_empty(self):
        df = pd.DataFrame({"if_pred": [0, 1]})
        assert get_anomaly_rate_by_step(df).empty

    def test_no_flag_column_returns_empty(self):
        df = pd.DataFrame({"process_step": ["Etching"], "some_col": [1]})
        assert get_anomaly_rate_by_step(df).empty

    def test_correct_columns(self):
        result = get_anomaly_rate_by_step(_make_scores())
        assert set(result.columns) == {"process_step", "total", "flagged", "rate_pct"}

    def test_rate_pct_calculation(self):
        df = pd.DataFrame({
            "process_step": ["Etching"] * 4,
            "ensemble_any": [1, 1, 0, 0],
        })
        result = get_anomaly_rate_by_step(df)
        assert result.iloc[0]["total"]    == 4
        assert result.iloc[0]["flagged"]  == 2
        assert result.iloc[0]["rate_pct"] == 50.0

    def test_falls_back_to_if_pred(self):
        df = pd.DataFrame({
            "process_step": ["CMP"] * 3,
            "if_pred":       [1, 0, 1],
        })
        result = get_anomaly_rate_by_step(df)
        assert result.iloc[0]["flagged"] == 2

    def test_sorted_by_rate_descending(self):
        scores = _make_scores(10)
        result = get_anomaly_rate_by_step(scores)
        rates = result["rate_pct"].tolist()
        assert rates == sorted(rates, reverse=True)

    def test_total_equals_row_count_per_step(self):
        scores = _make_scores(10)
        result = get_anomaly_rate_by_step(scores)
        for _, row in result.iterrows():
            expected = int((scores["process_step"] == row["process_step"]).sum())
            assert row["total"] == expected


# ══════════════════════════════════════════════════════════════════════════════
# TestGetAvailableCharts
# ══════════════════════════════════════════════════════════════════════════════

class TestGetAvailableCharts:
    def test_nonexistent_dir_returns_empty(self, tmp_path):
        result = get_available_charts(tmp_path / "no_such_dir")
        assert result == {}

    def test_empty_dir_returns_empty(self, tmp_path):
        assert get_available_charts(tmp_path) == {}

    def test_parses_double_underscore(self, tmp_path):
        (tmp_path / "Etching__gas_flow.png").touch()
        result = get_available_charts(tmp_path)
        assert "Etching | gas_flow" in result

    def test_returns_path_objects(self, tmp_path):
        (tmp_path / "Etching__temperature.png").touch()
        result = get_available_charts(tmp_path)
        for v in result.values():
            assert isinstance(v, Path)

    def test_fallback_for_files_without_dunder(self, tmp_path):
        (tmp_path / "overview.png").touch()
        result = get_available_charts(tmp_path)
        assert "overview" in result

    def test_multiple_charts(self, tmp_path):
        for name in ["Etching__temperature.png", "CMP__pressure.png",
                     "Deposition__gas_flow.png"]:
            (tmp_path / name).touch()
        result = get_available_charts(tmp_path)
        assert len(result) == 3
        assert "CMP | pressure" in result
        assert "Deposition | gas_flow" in result

    def test_only_png_files(self, tmp_path):
        (tmp_path / "Etching__temperature.png").touch()
        (tmp_path / "readme.txt").write_text("not a chart")
        result = get_available_charts(tmp_path)
        assert len(result) == 1


# ══════════════════════════════════════════════════════════════════════════════
# TestFormatRcaCandidates
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatRcaCandidates:
    def test_empty_list(self):
        assert format_rca_candidates([]) == []

    def test_rank_starts_at_one(self):
        cands = [_FakeCandidate("Etching", "high", ["rf_power"], ["SPC violation"])]
        result = format_rca_candidates(cands)
        assert result[0]["rank"] == 1

    def test_ranks_are_sequential(self):
        cands = [
            _FakeCandidate("Etching",    "high",   ["rf_power"],    ["ev1"]),
            _FakeCandidate("Deposition", "medium", ["temperature"], ["ev2"]),
        ]
        result = format_rca_candidates(cands)
        assert [r["rank"] for r in result] == [1, 2]

    def test_all_required_fields(self):
        cands = [_FakeCandidate("Etching", "high", ["rf_power"], ["ev"])]
        row = format_rca_candidates(cands)[0]
        for key in ("rank", "suspected_step", "confidence", "features", "evidence_count"):
            assert key in row, f"Missing key: {key}"

    def test_confidence_uppercased(self):
        cands = [_FakeCandidate("Etching", "high", [], [])]
        assert format_rca_candidates(cands)[0]["confidence"] == "HIGH"

    def test_no_features_returns_dash(self):
        cands = [_FakeCandidate("Etching", "low", [], [])]
        assert format_rca_candidates(cands)[0]["features"] == "—"

    def test_multiple_features_comma_joined(self):
        feats = ["rf_power", "gas_flow"]
        cands = [_FakeCandidate("Etching", "medium", feats, ["ev"])]
        row = format_rca_candidates(cands)[0]
        assert row["features"] == "rf_power, gas_flow"

    def test_evidence_count(self):
        cands = [_FakeCandidate("Etching", "high", [], ["ev1", "ev2", "ev3"])]
        assert format_rca_candidates(cands)[0]["evidence_count"] == 3

    def test_suspected_step_preserved(self):
        cands = [_FakeCandidate("CMP", "low", [], [])]
        assert format_rca_candidates(cands)[0]["suspected_step"] == "CMP"
