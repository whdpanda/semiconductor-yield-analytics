"""Tests for the synthetic process data generator (Module B).

These tests verify structure, correctness, and reproducibility of the generated data.
No real fab data is required — the generator is entirely self-contained.
"""

import numpy as np
import pandas as pd
import pytest

from semiconductor_yield.process.synthetic import (
    PROCESS_STEPS,
    ROOT_CAUSE_DESCRIPTIONS,
    STEP_NOMINALS,
    VALID_ANOMALY_TYPES,
    SyntheticProcessDataGenerator,
)

EXPECTED_COLUMNS = {
    "lot_id",
    "wafer_id",
    "timestamp",
    "process_step",
    "temperature",
    "pressure",
    "gas_flow",
    "rf_power",
    "exposure_dose",
    "film_thickness",
    "overlay_error",
    "defect_density",
    "yield_rate",
    "anomaly_label",
    "anomaly_type",
    "suspected_root_cause",
}


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """A small dataset for fast tests."""
    return SyntheticProcessDataGenerator(seed=42).generate(
        n_lots=10,
        n_wafers_per_lot=5,
        process_steps=PROCESS_STEPS,
        anomaly_rate=0.05,
    )


@pytest.fixture(scope="module")
def default_df() -> pd.DataFrame:
    """Full default dataset."""
    return SyntheticProcessDataGenerator(seed=42).generate()


# ── Structure ──────────────────────────────────────────────────────────────────

def test_returns_dataframe(small_df):
    assert isinstance(small_df, pd.DataFrame)


def test_expected_columns_present(small_df):
    missing = EXPECTED_COLUMNS - set(small_df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_no_extra_internal_columns(small_df):
    internal = {"lot_index", "wafer_index", "step_index"}
    present = internal & set(small_df.columns)
    assert not present, f"Internal columns leaked into output: {present}"


def test_row_count_matches_dimensions(small_df):
    expected = 10 * 5 * len(PROCESS_STEPS)
    assert len(small_df) == expected


def test_all_process_steps_present(small_df):
    assert set(small_df["process_step"].unique()) == set(PROCESS_STEPS)


def test_lot_ids_format(small_df):
    for lot_id in small_df["lot_id"].unique():
        assert lot_id.startswith("LOT_"), f"Unexpected lot_id format: {lot_id}"


def test_wafer_ids_format(small_df):
    for w_id in small_df["wafer_id"].unique():
        assert w_id.startswith("W_"), f"Unexpected wafer_id format: {w_id}"


def test_timestamp_is_datetime(small_df):
    assert pd.api.types.is_datetime64_any_dtype(small_df["timestamp"])


def test_timestamps_monotonically_nondecreasing(small_df):
    assert (small_df["timestamp"].diff().dropna() >= pd.Timedelta(0)).all()


# ── Anomaly labels ─────────────────────────────────────────────────────────────

def test_anomaly_label_is_bool(small_df):
    assert small_df["anomaly_label"].dtype == bool


def test_anomaly_type_values_valid(small_df):
    observed = set(small_df["anomaly_type"].unique())
    assert observed <= VALID_ANOMALY_TYPES, f"Unexpected types: {observed - VALID_ANOMALY_TYPES}"


def test_anomaly_label_consistent_with_type(small_df):
    is_normal = small_df["anomaly_type"] == "normal"
    assert (small_df.loc[is_normal, "anomaly_label"] == False).all()
    assert (small_df.loc[~is_normal, "anomaly_label"] == True).all()


def test_default_generation_has_some_anomalies(default_df):
    assert default_df["anomaly_label"].sum() > 0, "Expected at least one anomaly"


def test_all_four_anomaly_types_present_in_default(default_df):
    present = set(default_df["anomaly_type"].unique())
    for t in ["drift", "spike", "step_shift", "tool_offset"]:
        assert t in present, f"Anomaly type '{t}' not present in default generation"


# ── Numeric ranges ─────────────────────────────────────────────────────────────

def test_yield_rate_bounded(small_df):
    assert (small_df["yield_rate"] >= 0.0).all()
    assert (small_df["yield_rate"] <= 1.0).all()


def test_defect_density_nonnegative(small_df):
    assert (small_df["defect_density"] >= 0.0).all()


def test_temperature_plausible(small_df):
    # All step temperatures are between 10°C and 500°C under normal conditions.
    # Spikes can push this, so allow ±5σ headroom (worst case Deposition ~400±40°C → ~200-600).
    # A liberal bound: 0°C to 700°C
    assert (small_df["temperature"] > 0).all()
    assert (small_df["temperature"] < 700).all()


def test_nan_pattern_correct_for_exposure_dose(small_df):
    """exposure_dose should only be non-NaN for Lithography steps."""
    litho_mask = small_df["process_step"] == "Lithography"
    non_litho = small_df.loc[~litho_mask, "exposure_dose"]
    assert non_litho.isna().all(), "exposure_dose should be NaN for non-Lithography steps"

    litho_vals = small_df.loc[litho_mask, "exposure_dose"]
    assert litho_vals.notna().all(), "exposure_dose should be non-NaN for Lithography steps"


def test_nan_pattern_correct_for_overlay_error(small_df):
    """overlay_error is applicable only to Lithography and Metrology."""
    applicable = small_df["process_step"].isin(["Lithography", "Metrology"])
    assert small_df.loc[~applicable, "overlay_error"].isna().all()
    assert small_df.loc[applicable, "overlay_error"].notna().all()


def test_rf_power_nan_for_non_plasma_steps(small_df):
    """rf_power applies to Etching and Deposition only."""
    plasma = small_df["process_step"].isin(["Etching", "Deposition"])
    assert small_df.loc[~plasma, "rf_power"].isna().all()
    assert small_df.loc[plasma, "rf_power"].notna().all()


# ── Root cause ─────────────────────────────────────────────────────────────────

def test_suspected_root_cause_is_string(small_df):
    assert small_df["suspected_root_cause"].dtype == object
    assert small_df["suspected_root_cause"].notna().all()


def test_root_cause_matches_anomaly_type(small_df):
    for _, row in small_df.iterrows():
        expected = ROOT_CAUSE_DESCRIPTIONS[row["anomaly_type"]]
        assert row["suspected_root_cause"] == expected


# ── Reproducibility ────────────────────────────────────────────────────────────

def test_same_seed_produces_identical_output():
    gen1 = SyntheticProcessDataGenerator(seed=7)
    gen2 = SyntheticProcessDataGenerator(seed=7)
    df1 = gen1.generate(n_lots=5, n_wafers_per_lot=3)
    df2 = gen2.generate(n_lots=5, n_wafers_per_lot=3)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_output():
    gen1 = SyntheticProcessDataGenerator(seed=1)
    gen2 = SyntheticProcessDataGenerator(seed=2)
    df1 = gen1.generate(n_lots=10, n_wafers_per_lot=5)
    df2 = gen2.generate(n_lots=10, n_wafers_per_lot=5)
    assert not df1["temperature"].equals(df2["temperature"])


# ── Custom configuration ───────────────────────────────────────────────────────

def test_custom_process_steps():
    gen = SyntheticProcessDataGenerator(seed=42)
    df = gen.generate(n_lots=5, n_wafers_per_lot=3, process_steps=["Etching", "Deposition"])
    assert set(df["process_step"].unique()) == {"Etching", "Deposition"}
    assert len(df) == 5 * 3 * 2


def test_single_anomaly_type():
    gen = SyntheticProcessDataGenerator(seed=42)
    df = gen.generate(n_lots=20, n_wafers_per_lot=10, anomaly_types=["spike"])
    observed = set(df["anomaly_type"].unique()) - {"normal"}
    assert observed == {"spike"}, f"Expected only 'spike', got: {observed}"


def test_invalid_anomaly_type_raises():
    gen = SyntheticProcessDataGenerator(seed=42)
    with pytest.raises(ValueError, match="Unknown anomaly type"):
        gen.generate(anomaly_types=["nonexistent_type"])


def test_no_anomalies_when_empty_types():
    gen = SyntheticProcessDataGenerator(seed=42)
    df = gen.generate(n_lots=5, n_wafers_per_lot=3, anomaly_types=[])
    assert (df["anomaly_label"] == False).all()
    assert (df["anomaly_type"] == "normal").all()


# ── Step nominal values sanity check ──────────────────────────────────────────

def test_step_nominals_cover_all_steps():
    for step in PROCESS_STEPS:
        assert step in STEP_NOMINALS, f"No nominal defined for step: {step}"


def test_step_nominals_std_nonnegative():
    for step, params in STEP_NOMINALS.items():
        for param, (mean, std) in params.items():
            assert std >= 0, f"{step}/{param}: std must be >= 0, got {std}"
