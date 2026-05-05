"""Tests for SPC Western Electric Rules and visualization (Module B).

All tests use synthetic arrays — no external data files required.
matplotlib is set to the Agg backend via tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from semiconductor_yield.process.spc import (
    ControlLimits,
    SPCViolation,
    WE_RULES,
    check_western_electric_rules,
    compute_control_limits,
    run_spc,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def lim(mean: float = 0.0, std: float = 1.0) -> ControlLimits:
    return ControlLimits(mean=mean, std=std)


def ts(n: int) -> np.ndarray:
    return pd.date_range("2024-01-01", periods=n, freq="1h").values


def rule_violations(viols: list[SPCViolation], rule: str) -> list[SPCViolation]:
    return [v for v in viols if v.rule == rule]


# ── ControlLimits ──────────────────────────────────────────────────────────────

def test_control_limits_ucl():
    assert ControlLimits(mean=10.0, std=2.0).ucl == pytest.approx(16.0)


def test_control_limits_lcl():
    assert ControlLimits(mean=10.0, std=2.0).lcl == pytest.approx(4.0)


def test_control_limits_2sigma_zones():
    cl = ControlLimits(mean=0.0, std=1.0)
    assert cl.ucl_2 == pytest.approx(2.0)
    assert cl.lcl_2 == pytest.approx(-2.0)


def test_control_limits_1sigma_zones():
    cl = ControlLimits(mean=0.0, std=1.0)
    assert cl.ucl_1 == pytest.approx(1.0)
    assert cl.lcl_1 == pytest.approx(-1.0)


def test_control_limits_asymmetric_mean():
    cl = ControlLimits(mean=100.0, std=5.0)
    assert cl.ucl == pytest.approx(115.0)
    assert cl.lcl == pytest.approx(85.0)


# ── compute_control_limits ─────────────────────────────────────────────────────

def test_compute_control_limits_mean():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert compute_control_limits(values).mean == pytest.approx(2.0)


def test_compute_control_limits_std():
    values = np.array([0.0, 2.0])  # mean=1, std=√2 ≈ 1.414
    cl = compute_control_limits(values)
    assert cl.std == pytest.approx(np.std([0.0, 2.0], ddof=1))


def test_compute_control_limits_ignores_nan():
    values = np.array([0.0, np.nan, 2.0, np.nan, 4.0])
    assert compute_control_limits(values).mean == pytest.approx(2.0)


def test_compute_control_limits_raises_if_one_value():
    with pytest.raises(ValueError, match="at least 2"):
        compute_control_limits(np.array([1.0]))


def test_compute_control_limits_raises_if_all_nan():
    with pytest.raises(ValueError, match="at least 2"):
        compute_control_limits(np.array([np.nan, np.nan]))


# ── Rule 1: one point beyond 3σ ───────────────────────────────────────────────

def test_rule1_above_3sigma_triggers():
    values = np.array([0.0, 0.0, 0.0, 3.5])  # index 3 is 3.5σ above
    viols = check_western_electric_rules(values, ts(4), lim(), "x")
    r1 = rule_violations(viols, "Rule 1")
    assert len(r1) == 1
    assert r1[0].index == 3


def test_rule1_below_3sigma_triggers():
    values = np.array([0.0, 0.0, 0.0, -3.5])
    viols = check_western_electric_rules(values, ts(4), lim(), "x")
    assert len(rule_violations(viols, "Rule 1")) == 1


def test_rule1_exactly_at_3sigma_does_not_trigger():
    # UCL is mean + 3*std; the rule is strictly beyond (>) not at
    values = np.array([0.0, 0.0, 0.0, 3.0])
    viols = check_western_electric_rules(values, ts(4), lim(), "x")
    assert len(rule_violations(viols, "Rule 1")) == 0


def test_rule1_multiple_violations():
    values = np.array([4.0, 0.0, -4.0, 0.0])
    viols = check_western_electric_rules(values, ts(4), lim(), "x")
    assert len(rule_violations(viols, "Rule 1")) == 2


def test_rule1_severity_is_high():
    values = np.array([0.0, 4.0])
    viols = check_western_electric_rules(values, ts(2), lim(), "x")
    r1 = rule_violations(viols, "Rule 1")
    assert r1[0].severity == "HIGH"


def test_rule1_normal_tight_data_no_violation():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.5, 200)  # all within ±1.5σ relative to lim(std=1)
    viols = check_western_electric_rules(values, ts(200), lim(), "x")
    assert len(rule_violations(viols, "Rule 1")) == 0


# ── Rule 2: 2 of 3 consecutive beyond 2σ on same side ────────────────────────

def test_rule2_two_above_2sigma_triggers():
    # Window at i=4: [-0.1, 2.5, 2.5] → 2 above ucl_2=2.0
    values = np.array([0.0, 0.0, -0.1, 2.5, 2.5])
    viols = check_western_electric_rules(values, ts(5), lim(), "x")
    assert len(rule_violations(viols, "Rule 2")) >= 1


def test_rule2_two_below_2sigma_triggers():
    values = np.array([0.0, 0.0, 0.1, -2.5, -2.5])
    viols = check_western_electric_rules(values, ts(5), lim(), "x")
    assert len(rule_violations(viols, "Rule 2")) >= 1


def test_rule2_opposite_sides_does_not_trigger():
    # One above 2σ, one below 2σ in the same window — mixed sides
    values = np.array([0.0, 0.0, 2.5, 0.0, -2.5])
    viols = check_western_electric_rules(values, ts(5), lim(), "x")
    assert len(rule_violations(viols, "Rule 2")) == 0


def test_rule2_one_of_three_does_not_trigger():
    # Only the last point beyond 2σ — not enough
    values = np.array([0.0, 0.0, 0.5, 0.3, 2.5])
    viols = check_western_electric_rules(values, ts(5), lim(), "x")
    assert len(rule_violations(viols, "Rule 2")) == 0


def test_rule2_severity_is_medium():
    values = np.array([0.0, 0.0, -0.1, 2.5, 2.5])
    viols = check_western_electric_rules(values, ts(5), lim(), "x")
    r2 = rule_violations(viols, "Rule 2")
    assert all(v.severity == "MEDIUM" for v in r2)


def test_rule2_requires_at_least_3_points():
    # Only 2 points — Rule 2 window never fills
    values = np.array([2.5, 2.5])
    viols = check_western_electric_rules(values, ts(2), lim(), "x")
    assert len(rule_violations(viols, "Rule 2")) == 0


# ── Rule 3: 4 of 5 consecutive beyond 1σ on same side ────────────────────────

def test_rule3_four_above_1sigma_triggers():
    # Window at i=5: [0.5, 1.5, 1.5, 1.5, 1.5] → 4 above ucl_1=1.0
    values = np.array([0.0, 0.5, 1.5, 1.5, 1.5, 1.5])
    viols = check_western_electric_rules(values, ts(6), lim(), "x")
    assert len(rule_violations(viols, "Rule 3")) >= 1


def test_rule3_four_below_1sigma_triggers():
    values = np.array([0.0, -0.5, -1.5, -1.5, -1.5, -1.5])
    viols = check_western_electric_rules(values, ts(6), lim(), "x")
    assert len(rule_violations(viols, "Rule 3")) >= 1


def test_rule3_mixed_sides_does_not_trigger():
    # 2 above 1σ, 2 below 1σ in a window
    values = np.array([0.0, 1.5, 1.5, -1.5, -1.5, 1.5])
    viols = check_western_electric_rules(values, ts(6), lim(), "x")
    assert len(rule_violations(viols, "Rule 3")) == 0


def test_rule3_three_of_five_does_not_trigger():
    values = np.array([0.0, 0.0, 1.5, 1.5, 1.5, 0.5])  # only 3 above 1σ
    viols = check_western_electric_rules(values, ts(6), lim(), "x")
    assert len(rule_violations(viols, "Rule 3")) == 0


def test_rule3_severity_is_low():
    values = np.array([0.0, 0.5, 1.5, 1.5, 1.5, 1.5])
    viols = check_western_electric_rules(values, ts(6), lim(), "x")
    r3 = rule_violations(viols, "Rule 3")
    assert all(v.severity == "LOW" for v in r3)


def test_rule3_requires_at_least_5_points():
    values = np.array([1.5, 1.5, 1.5, 1.5])  # 4 points — window never fills
    viols = check_western_electric_rules(values, ts(4), lim(), "x")
    assert len(rule_violations(viols, "Rule 3")) == 0


# ── Rule 4: 8 consecutive on same side of center line ─────────────────────────

def test_rule4_eight_consecutive_above_mean_triggers():
    # index 0 = 0.0 (at mean, neither side); indices 1-8 all positive
    values = np.array([0.0, 0.5, 0.1, 0.3, 0.2, 0.4, 0.1, 0.3, 0.5])
    viols = check_western_electric_rules(values, ts(9), lim(), "x")
    assert len(rule_violations(viols, "Rule 4")) >= 1


def test_rule4_eight_consecutive_below_mean_triggers():
    values = np.array([0.0, -0.5, -0.1, -0.3, -0.2, -0.4, -0.1, -0.3, -0.5])
    viols = check_western_electric_rules(values, ts(9), lim(), "x")
    assert len(rule_violations(viols, "Rule 4")) >= 1


def test_rule4_seven_consecutive_does_not_trigger():
    # indices 1-7 positive, but index 8 breaks the run
    values = np.array([-0.5, 0.5, 0.1, 0.3, 0.2, 0.4, 0.1, 0.3, -0.5])
    viols = check_western_electric_rules(values, ts(9), lim(), "x")
    assert len(rule_violations(viols, "Rule 4")) == 0


def test_rule4_alternating_sides_does_not_trigger():
    values = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    viols = check_western_electric_rules(values, ts(9), lim(), "x")
    assert len(rule_violations(viols, "Rule 4")) == 0


def test_rule4_point_at_mean_breaks_run():
    # First 7 are positive, then exactly at mean (0.0), then positive again
    values = np.array([0.3, 0.2, 0.4, 0.1, 0.3, 0.5, 0.2, 0.0, 0.4])
    viols = check_western_electric_rules(values, ts(9), lim(), "x")
    assert len(rule_violations(viols, "Rule 4")) == 0


def test_rule4_severity_is_low():
    values = np.array([0.0, 0.5, 0.1, 0.3, 0.2, 0.4, 0.1, 0.3, 0.5])
    viols = check_western_electric_rules(values, ts(9), lim(), "x")
    r4 = rule_violations(viols, "Rule 4")
    assert all(v.severity == "LOW" for v in r4)


def test_rule4_requires_at_least_8_points():
    values = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # 7 points
    viols = check_western_electric_rules(values, ts(7), lim(), "x")
    assert len(rule_violations(viols, "Rule 4")) == 0


# ── No violations for in-control process ──────────────────────────────────────

def test_no_violations_for_tightly_controlled_process():
    rng = np.random.default_rng(42)
    # All values within ±0.7σ — cannot trigger Rules 1, 2, 3 against lim(std=1)
    values = rng.uniform(-0.7, 0.7, 200)
    viols = check_western_electric_rules(values, ts(200), lim(), "x")
    assert len(rule_violations(viols, "Rule 1")) == 0
    assert len(rule_violations(viols, "Rule 2")) == 0


# ── SPCViolation fields ────────────────────────────────────────────────────────

def test_violation_carries_correct_feature_name():
    values = np.array([0.0, 4.0])
    viols = check_western_electric_rules(values, ts(2), lim(), "temperature")
    assert viols[0].feature == "temperature"


def test_violation_carries_correct_rule_description():
    values = np.array([0.0, 4.0])
    viols = check_western_electric_rules(values, ts(2), lim(), "x")
    r1 = rule_violations(viols, "Rule 1")
    assert r1[0].rule_description == WE_RULES["Rule 1"]


def test_violation_timestamp_is_pandas_timestamp():
    values = np.array([0.0, 4.0])
    viols = check_western_electric_rules(values, ts(2), lim(), "x")
    assert isinstance(viols[0].timestamp, pd.Timestamp)


def test_violation_value_matches_data():
    values = np.array([0.0, 5.0])
    viols = check_western_electric_rules(values, ts(2), lim(), "x")
    r1 = rule_violations(viols, "Rule 1")
    assert r1[0].value == pytest.approx(5.0)


# ── run_spc ────────────────────────────────────────────────────────────────────

def _make_df(step: str, feature: str, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(values), freq="h"),
            "process_step": step,
            feature: values,
        }
    )


def test_run_spc_returns_dataframe_and_dict():
    df = _make_df("Etching", "temperature", np.zeros(20))
    viols_df, limits_map = run_spc(df, feature_cols=["temperature"])
    assert isinstance(viols_df, pd.DataFrame)
    assert isinstance(limits_map, dict)


def test_run_spc_violations_df_required_columns():
    values = np.concatenate([np.zeros(15), np.full(5, 10.0)])  # large step
    df = _make_df("Etching", "temperature", values)
    viols_df, _ = run_spc(df, feature_cols=["temperature"])
    required = {"timestamp", "feature", "process_step", "rule", "severity", "series_index"}
    assert required.issubset(set(viols_df.columns))


def test_run_spc_skips_all_nan_feature():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "process_step": "Lithography",
            "rf_power": float("nan"),  # not applicable for this step
            "temperature": np.random.default_rng(0).normal(23, 0.3, 10),
        }
    )
    _, limits_map = run_spc(df, feature_cols=["rf_power", "temperature"])
    assert all(feat != "rf_power" for feat, _ in limits_map)


def test_run_spc_limits_map_keyed_by_feature_and_step():
    df = _make_df("CMP", "pressure", np.random.default_rng(0).normal(3000, 100, 30))
    _, limits_map = run_spc(df, feature_cols=["pressure"])
    assert ("pressure", "CMP") in limits_map


def test_run_spc_limits_map_values_are_control_limits():
    df = _make_df("CMP", "pressure", np.random.default_rng(0).normal(3000, 100, 30))
    _, limits_map = run_spc(df, feature_cols=["pressure"])
    cl = limits_map[("pressure", "CMP")]
    assert isinstance(cl, ControlLimits)
    assert cl.mean == pytest.approx(3000.0, abs=100.0)


def test_run_spc_empty_df_for_in_control_process():
    rng = np.random.default_rng(0)
    df = _make_df("Etching", "temperature", rng.normal(60.0, 0.05, 100))
    viols_df, _ = run_spc(df, feature_cols=["temperature"])
    # With extremely tight spread (0.05 °C std), Rule 1 cannot fire
    assert len(viols_df[viols_df["rule"] == "Rule 1"]) == 0


def test_run_spc_no_group_col():
    df = _make_df("Etching", "temperature", np.random.default_rng(0).normal(0, 1, 30))
    viols_df, limits_map = run_spc(df, feature_cols=["temperature"], group_col=None)
    assert isinstance(viols_df, pd.DataFrame)
    assert ("temperature", "all") in limits_map


def test_run_spc_multi_step_independent_limits():
    df1 = _make_df("Etching", "temperature", np.full(20, 60.0) + np.random.default_rng(0).normal(0, 3, 20))
    df2 = _make_df("CMP", "temperature", np.full(20, 25.0) + np.random.default_rng(1).normal(0, 2, 20))
    df = pd.concat([df1, df2], ignore_index=True)
    _, limits_map = run_spc(df, feature_cols=["temperature"])
    assert ("temperature", "Etching") in limits_map
    assert ("temperature", "CMP") in limits_map
    assert limits_map[("temperature", "Etching")].mean != limits_map[("temperature", "CMP")].mean


def test_run_spc_violations_sorted_by_timestamp():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "process_step": "Etching",
            "temperature": [0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
        }
    )
    viols_df, _ = run_spc(df, feature_cols=["temperature"])
    if len(viols_df) > 1:
        ts_vals = viols_df["timestamp"].values
        assert (ts_vals[1:] >= ts_vals[:-1]).all()


# ── Visualization ──────────────────────────────────────────────────────────────

def test_plot_control_chart_creates_file(tmp_path):
    from semiconductor_yield.process.visualization import plot_control_chart

    cl = ControlLimits(mean=0.0, std=1.0)
    values = np.random.default_rng(0).normal(0, 1, 50)
    empty_viols = pd.DataFrame(columns=["series_index", "value"])
    out = tmp_path / "chart.png"
    plot_control_chart(values, cl, empty_viols, "temperature", "Etching", out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_control_chart_with_violations(tmp_path):
    from semiconductor_yield.process.visualization import plot_control_chart

    cl = ControlLimits(mean=0.0, std=1.0)
    values = np.array([0.0, 0.0, 0.0, 4.5, 0.0, 0.0])
    viols = pd.DataFrame({"series_index": [3], "value": [4.5]})
    out = tmp_path / "chart_with_viols.png"
    plot_control_chart(values, cl, viols, "temperature", "Etching", out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_control_chart_creates_parent_dir(tmp_path):
    from semiconductor_yield.process.visualization import plot_control_chart

    cl = ControlLimits(mean=3000.0, std=100.0)
    values = np.ones(20) * 3000.0
    empty_viols = pd.DataFrame(columns=["series_index", "value"])
    out = tmp_path / "nested" / "subdir" / "chart.png"
    plot_control_chart(values, cl, empty_viols, "pressure", "CMP", out)
    assert out.exists()


def test_plot_spc_summary_creates_files(tmp_path):
    from semiconductor_yield.process.visualization import plot_spc_summary

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h"),
            "process_step": "Etching",
            "temperature": rng.normal(60.0, 3.0, 30),
        }
    )
    _, limits_map = run_spc(df, feature_cols=["temperature"])
    viols_df, _ = run_spc(df, feature_cols=["temperature"])

    created = plot_spc_summary(df, limits_map, viols_df, output_dir=tmp_path)
    assert len(created) > 0
    for path in created.values():
        assert path.exists()
        assert path.suffix == ".png"
