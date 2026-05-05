"""Tests for process RCA candidate analysis (rca.py and report.py).

All tests use in-process synthetic DataFrames. No external files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from semiconductor_yield.process.rca import (
    KNOWN_SYNTHETIC_STEPS,
    LIMITATION_NOTE,
    RCACandidate,
    analyze,
)
from semiconductor_yield.process.report import generate_markdown_report


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def base_df() -> pd.DataFrame:
    """Small synthetic process DataFrame with three steps, no anomalies."""
    rng = np.random.default_rng(0)
    n_per_step = 30

    def _step_block(step: str, start: int) -> dict:
        return {
            "lot_id":       [f"LOT_{(start + i) // 5:03d}" for i in range(n_per_step)],
            "wafer_id":     [f"W_{i % 5:02d}" for i in range(n_per_step)],
            "process_step": [step] * n_per_step,
            "timestamp":    pd.date_range("2024-01-01", periods=n_per_step, freq="h")[
                                pd.Index(range(n_per_step))
                            ],
        }

    rows_etch = _step_block("Etching", 0)
    rows_lith = _step_block("Lithography", n_per_step)
    rows_cmp  = _step_block("CMP", 2 * n_per_step)

    common = {"temperature": rng.normal(60, 2, n_per_step),
              "pressure":    rng.normal(10, 0.5, n_per_step)}

    df_etch = pd.DataFrame({**rows_etch, **common,
                            "gas_flow": rng.normal(100, 3, n_per_step),
                            "rf_power": rng.normal(500, 10, n_per_step),
                            "film_thickness": rng.normal(1000, 30, n_per_step),
                            "defect_density": rng.uniform(0.02, 0.06, n_per_step),
                            "anomaly_label": np.zeros(n_per_step, dtype=bool)})

    df_lith = pd.DataFrame({**rows_lith, **common,
                            "exposure_dose": rng.normal(25, 0.2, n_per_step),
                            "overlay_error": rng.normal(5, 1, n_per_step),
                            "defect_density": rng.uniform(0.01, 0.03, n_per_step),
                            "anomaly_label": np.zeros(n_per_step, dtype=bool)})

    df_cmp = pd.DataFrame({**rows_cmp, **common,
                           "film_thickness": rng.normal(500, 20, n_per_step),
                           "defect_density": rng.uniform(0.02, 0.05, n_per_step),
                           "anomaly_label": np.zeros(n_per_step, dtype=bool)})

    return pd.concat([df_etch, df_lith, df_cmp], ignore_index=True)


@pytest.fixture
def etching_spc_violations() -> pd.DataFrame:
    """SPC violation rows for rf_power and gas_flow at Etching step."""
    rows = []
    for i in range(8):
        rows.append({
            "timestamp":       pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 3),
            "feature":         "rf_power",
            "process_step":    "Etching",
            "rule":            "Rule 1",
            "rule_description":"One point beyond 3σ",
            "value":           570.0 + i * 2,
            "severity":        "HIGH",
            "series_index":    i,
        })
    for i in range(4):
        rows.append({
            "timestamp":       pd.Timestamp("2024-01-02") + pd.Timedelta(hours=i * 5),
            "feature":         "gas_flow",
            "process_step":    "Etching",
            "rule":            "Rule 2",
            "rule_description":"Two of three consecutive points beyond 2σ on same side",
            "value":           115.0 + i,
            "severity":        "MEDIUM",
            "series_index":    i,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def etching_anomaly_scores(base_df) -> pd.DataFrame:
    """Anomaly scores with high flagging rate at Etching, normal elsewhere."""
    rng = np.random.default_rng(42)
    df = base_df[["lot_id", "wafer_id", "process_step", "timestamp", "anomaly_label"]].copy()
    df["if_score"] = rng.uniform(0.0, 0.05, len(df))
    df["ae_score"] = rng.uniform(0.0, 0.01, len(df))
    df["if_pred"] = 0
    df["ae_pred"] = 0
    df["ensemble_any"] = 0
    df["ensemble_all"] = 0
    # Flag ~40% of Etching rows
    etch_mask = df["process_step"] == "Etching"
    etch_idx = df.index[etch_mask]
    flagged = rng.choice(etch_idx, size=12, replace=False)
    df.loc[flagged, ["if_pred", "ae_pred", "ensemble_any", "ensemble_all"]] = 1
    return df


@pytest.fixture
def single_feature_spc() -> pd.DataFrame:
    """SPC violations for only one feature at one step."""
    return pd.DataFrame([{
        "timestamp":    pd.Timestamp("2024-01-01"),
        "feature":      "temperature",
        "process_step": "CMP",
        "rule":         "Rule 4",
        "rule_description": "Eight consecutive points on same side of center line",
        "value":        27.0,
        "severity":     "LOW",
        "series_index": 7,
    }])


# ── Core analysis tests ────────────────────────────────────────────────────────

class TestRCAAnalyze:
    def test_returns_list_of_rca_candidates(self, base_df, etching_spc_violations):
        result = analyze(base_df, spc_violations=etching_spc_violations)
        assert isinstance(result, list)
        assert all(isinstance(c, RCACandidate) for c in result)

    def test_etching_anomaly_ranks_etching_first(
        self, base_df, etching_spc_violations, etching_anomaly_scores
    ):
        """When rf_power and gas_flow at Etching have multiple SPC violations plus
        high anomaly detection rate, Etching must be the top-ranked candidate."""
        candidates = analyze(
            base_df,
            spc_violations=etching_spc_violations,
            anomaly_scores=etching_anomaly_scores,
        )
        assert len(candidates) > 0, "Expected at least one candidate"
        assert candidates[0].suspected_process_step == "Etching", (
            f"Expected Etching as top candidate, got {candidates[0].suspected_process_step}"
        )

    def test_etching_suspicious_features_include_discriminating_features(
        self, base_df, etching_spc_violations
    ):
        """rf_power and gas_flow are Etching-discriminating; they must appear."""
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        etching = next((c for c in candidates if c.suspected_process_step == "Etching"), None)
        assert etching is not None
        assert "rf_power" in etching.suspicious_features
        assert "gas_flow" in etching.suspicious_features

    def test_top_n_limits_output_length(self, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations, top_n=1)
        assert len(candidates) <= 1

    def test_no_evidence_returns_empty_list(self, base_df):
        """With neither SPC violations nor anomaly scores, no candidates returned."""
        candidates = analyze(base_df, spc_violations=None, anomaly_scores=None)
        assert candidates == []

    def test_only_anomaly_scores_produces_candidates(
        self, base_df, etching_anomaly_scores
    ):
        candidates = analyze(base_df, anomaly_scores=etching_anomaly_scores)
        assert len(candidates) > 0


class TestConfidenceLevel:
    def test_confidence_low_for_single_feature(self, base_df, single_feature_spc):
        candidates = analyze(base_df, spc_violations=single_feature_spc)
        assert len(candidates) > 0
        top = candidates[0]
        assert top.confidence_level == "low", (
            f"Single-feature anomaly should yield low confidence, got {top.confidence_level!r}"
        )

    def test_confidence_medium_or_high_for_two_features(
        self, base_df, etching_spc_violations
    ):
        """rf_power + gas_flow = 2 features; should yield at least medium."""
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        etching = next(c for c in candidates if c.suspected_process_step == "Etching")
        assert etching.confidence_level in ("medium", "high")

    def test_high_confidence_with_corroborating_anomaly_scores(
        self, base_df, etching_spc_violations, etching_anomaly_scores
    ):
        """SPC + anomaly detector agreement on Etching should raise confidence."""
        candidates = analyze(
            base_df,
            spc_violations=etching_spc_violations,
            anomaly_scores=etching_anomaly_scores,
        )
        etching = next(c for c in candidates if c.suspected_process_step == "Etching")
        assert etching.confidence_level in ("medium", "high")

    def test_confidence_values_are_valid_literals(self, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        for c in candidates:
            assert c.confidence_level in ("low", "medium", "high")


class TestLimitationNote:
    def test_limitation_note_always_present_and_nonempty(
        self, base_df, etching_spc_violations
    ):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c.limitation_note, str)
            assert len(c.limitation_note) > 20, "limitation_note must be a substantive string"

    def test_limitation_note_contains_candidate_language(
        self, base_df, etching_spc_violations
    ):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        for c in candidates:
            lower = c.limitation_note.lower()
            assert "candidate" in lower or "not a confirmed" in lower, (
                f"limitation_note must use candidate language, got: {c.limitation_note!r}"
            )

    def test_limitation_note_module_constant_is_nonempty(self):
        assert isinstance(LIMITATION_NOTE, str)
        assert len(LIMITATION_NOTE) > 50


class TestSECOMAnonymousMode:
    @pytest.fixture
    def secom_df(self) -> pd.DataFrame:
        """Anonymous feature DataFrame mimicking SECOM structure."""
        rng = np.random.default_rng(7)
        n = 60
        return pd.DataFrame({
            "lot_id":    [f"LOT_{i // 5:03d}" for i in range(n)],
            "process_step": ["anonymous_step"] * n,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            **{f"Feature_{j:03d}": rng.normal(0, 1, n) for j in range(10)},
        })

    @pytest.fixture
    def secom_spc_violations(self) -> pd.DataFrame:
        """SPC violations with anonymous feature names."""
        return pd.DataFrame([
            {
                "timestamp":    pd.Timestamp("2024-01-01"),
                "feature":      f"Feature_{j:03d}",
                "process_step": "anonymous_step",
                "rule":         "Rule 1",
                "rule_description": "One point beyond 3σ",
                "value":        4.2,
                "severity":     "HIGH",
                "series_index": 0,
            }
            for j in range(5)
        ])

    def test_secom_mode_no_known_fab_step_names_in_candidates(
        self, secom_df, secom_spc_violations
    ):
        """No candidate.suspected_process_step must be a known synthetic step name."""
        candidates = analyze(
            secom_df,
            spc_violations=secom_spc_violations,
            data_source="secom",
        )
        assert len(candidates) > 0
        for c in candidates:
            assert c.suspected_process_step not in KNOWN_SYNTHETIC_STEPS, (
                f"SECOM mode must not output known synthetic step: "
                f"{c.suspected_process_step!r}"
            )

    def test_secom_mode_no_known_fab_step_in_any_field(
        self, secom_df, secom_spc_violations
    ):
        """Check evidence and recommended_checks text also avoid fab step names."""
        candidates = analyze(
            secom_df,
            spc_violations=secom_spc_violations,
            data_source="secom",
        )
        for c in candidates:
            full_text = " ".join(c.evidence + c.recommended_checks)
            for step_name in KNOWN_SYNTHETIC_STEPS:
                assert step_name not in full_text, (
                    f"Known fab step {step_name!r} found in SECOM candidate output"
                )

    def test_secom_mode_limitation_note_present(self, secom_df, secom_spc_violations):
        candidates = analyze(
            secom_df,
            spc_violations=secom_spc_violations,
            data_source="secom",
        )
        for c in candidates:
            assert c.limitation_note, "limitation_note must be present in SECOM mode"
            assert "anonymous" in c.limitation_note.lower() or "candidate" in c.limitation_note.lower()

    def test_secom_mode_step_label_uses_anonymous_prefix(
        self, secom_df, secom_spc_violations
    ):
        candidates = analyze(
            secom_df,
            spc_violations=secom_spc_violations,
            data_source="secom",
        )
        for c in candidates:
            assert "anonymous" in c.suspected_process_step.lower(), (
                f"SECOM candidate step label should use 'anonymous' prefix, "
                f"got {c.suspected_process_step!r}"
            )


class TestRCAOutputStructure:
    def test_evidence_is_nonempty_list(self, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        for c in candidates:
            assert isinstance(c.evidence, list)
            assert len(c.evidence) > 0

    def test_recommended_checks_is_nonempty_list(self, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        for c in candidates:
            assert isinstance(c.recommended_checks, list)
            assert len(c.recommended_checks) > 0

    def test_suspicious_features_is_list(self, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        for c in candidates:
            assert isinstance(c.suspicious_features, list)

    def test_suspected_process_step_is_str(self, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        for c in candidates:
            assert isinstance(c.suspected_process_step, str)
            assert len(c.suspected_process_step) > 0

    def test_only_anomaly_evidence_has_correct_step(
        self, base_df, etching_anomaly_scores
    ):
        """When only anomaly scores are given, Etching should still be a candidate."""
        candidates = analyze(base_df, anomaly_scores=etching_anomaly_scores)
        step_names = [c.suspected_process_step for c in candidates]
        assert "Etching" in step_names


# ── Report generation tests ────────────────────────────────────────────────────

class TestGenerateMarkdownReport:
    def test_creates_file(self, tmp_path, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        out = tmp_path / "rca_report.md"
        generate_markdown_report(candidates, data_source="synthetic", output_path=out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        out = tmp_path / "nested" / "deep" / "rca.md"
        generate_markdown_report(candidates, data_source="synthetic", output_path=out)
        assert out.exists()

    def test_report_contains_disclaimer(self, tmp_path, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        out = tmp_path / "rca.md"
        text = generate_markdown_report(candidates, data_source="synthetic", output_path=out)
        assert "Disclaimer" in text or "disclaimer" in text.lower()

    def test_report_contains_candidate_language(
        self, tmp_path, base_df, etching_spc_violations
    ):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        out = tmp_path / "rca.md"
        text = generate_markdown_report(candidates, data_source="synthetic", output_path=out)
        assert "candidate" in text.lower() or "CANDIDATE" in text

    def test_report_contains_etching(self, tmp_path, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        out = tmp_path / "rca.md"
        text = generate_markdown_report(candidates, data_source="synthetic", output_path=out)
        assert "Etching" in text

    def test_report_returns_str(self, tmp_path, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        out = tmp_path / "rca.md"
        result = generate_markdown_report(candidates, data_source="synthetic", output_path=out)
        assert isinstance(result, str)

    def test_empty_candidates_report_still_generated(self, tmp_path, base_df):
        out = tmp_path / "rca_empty.md"
        text = generate_markdown_report([], data_source="synthetic", output_path=out)
        assert out.exists()
        assert "No root cause candidates" in text or len(text) > 50

    def test_report_includes_meta_if_provided(self, tmp_path, base_df, etching_spc_violations):
        candidates = analyze(base_df, spc_violations=etching_spc_violations)
        out = tmp_path / "rca_meta.md"
        text = generate_markdown_report(
            candidates, data_source="synthetic", output_path=out,
            meta={"n_samples": 1000, "feature_set": "full"},
        )
        assert "1,000" in text or "1000" in text
        assert "full" in text
