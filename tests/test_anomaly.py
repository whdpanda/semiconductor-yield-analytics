"""Tests for process anomaly detection — features.py and anomaly.py.

All tests use small synthetic arrays. No external data files required.
PyTorch is imported lazily inside AutoencoderDetector, so the import of
this test module itself does not trigger a torch import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from semiconductor_yield.process.features import (
    ANOMALY_INPUT_FEATURES,
    FEATURE_SET_MAP,
    LABEL_COLS,
    PROCESS_ONLY_FEATURES,
    PROCESS_PLUS_METROLOGY_FEATURES,
    SENSOR_FEATURES,
    YIELD_FEATURES,
    get_feature_matrix,
    lot_split,
)
from semiconductor_yield.process.anomaly import (
    AutoencoderDetector,
    IsolationForestDetector,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """60-row DataFrame with realistic NaN pattern (step-specific parameters)."""
    rng = np.random.default_rng(0)
    n = 60
    nan = float("nan")
    return pd.DataFrame(
        {
            "lot_id": [f"LOT_{i//5:03d}" for i in range(n)],
            "wafer_id": [f"W_{i%5:02d}" for i in range(n)],
            "process_step": ["Etching"] * 20 + ["Lithography"] * 20 + ["CMP"] * 20,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "temperature": rng.normal(60, 3, n),
            "pressure": rng.normal(10, 1, n),
            "gas_flow": np.concatenate(
                [rng.normal(100, 5, 20), np.full(20, nan), np.full(20, nan)]
            ),
            "rf_power": np.concatenate(
                [rng.normal(500, 15, 20), np.full(20, nan), np.full(20, nan)]
            ),
            "exposure_dose": np.concatenate(
                [np.full(20, nan), rng.normal(25, 0.3, 20), np.full(20, nan)]
            ),
            "film_thickness": np.concatenate(
                [rng.normal(1000, 40, 20), np.full(20, nan), rng.normal(500, 25, 20)]
            ),
            "overlay_error": np.concatenate(
                [np.full(20, nan), rng.normal(5, 1.5, 20), np.full(20, nan)]
            ),
            "defect_density": np.abs(rng.normal(0.03, 0.01, n)),
            "yield_rate": rng.uniform(0.85, 1.0, n),
            "anomaly_label": np.zeros(n, dtype=bool),
        }
    )


@pytest.fixture(scope="module")
def feature_matrix(small_df) -> tuple[np.ndarray, list[str]]:
    return get_feature_matrix(small_df)


@pytest.fixture(scope="module")
def fitted_if(feature_matrix):
    X, cols = feature_matrix
    det = IsolationForestDetector(n_estimators=10, random_state=0)
    det.fit(X, cols)
    return det


@pytest.fixture(scope="module")
def fitted_ae(feature_matrix):
    X, cols = feature_matrix
    det = AutoencoderDetector(hidden_dims=(8, 4), epochs=2, batch_size=16)
    det.fit(X, cols)
    return det


# ── lot_split ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def split_dfs(small_df):
    """Return (train_df, val_df, test_df) from small_df using default fracs."""
    return lot_split(small_df)


class TestLotSplit:
    def test_no_train_test_lot_overlap(self, split_dfs):
        train_df, _, test_df = split_dfs
        assert set(train_df["lot_id"]).isdisjoint(set(test_df["lot_id"]))

    def test_no_train_val_lot_overlap(self, split_dfs):
        train_df, val_df, _ = split_dfs
        assert set(train_df["lot_id"]).isdisjoint(set(val_df["lot_id"]))

    def test_no_val_test_lot_overlap(self, split_dfs):
        _, val_df, test_df = split_dfs
        assert set(val_df["lot_id"]).isdisjoint(set(test_df["lot_id"]))

    def test_all_lots_accounted_for(self, small_df, split_dfs):
        train_df, val_df, test_df = split_dfs
        all_split = set(train_df["lot_id"]) | set(val_df["lot_id"]) | set(test_df["lot_id"])
        assert all_split == set(small_df["lot_id"])

    def test_all_rows_accounted_for(self, small_df, split_dfs):
        train_df, val_df, test_df = split_dfs
        assert len(train_df) + len(val_df) + len(test_df) == len(small_df)

    def test_train_is_largest_split(self, split_dfs):
        train_df, val_df, test_df = split_dfs
        assert len(train_df) >= len(val_df)
        assert len(train_df) >= len(test_df)

    def test_each_split_is_nonempty(self, split_dfs):
        for df in split_dfs:
            assert len(df) > 0

    def test_reproducible_with_same_seed(self, small_df):
        a_train, a_val, a_test = lot_split(small_df, random_state=7)
        b_train, b_val, b_test = lot_split(small_df, random_state=7)
        assert set(a_train["lot_id"]) == set(b_train["lot_id"])
        assert set(a_test["lot_id"]) == set(b_test["lot_id"])

    def test_different_seed_produces_different_split(self, small_df):
        lots_a = set(lot_split(small_df, random_state=1)[2]["lot_id"])
        lots_b = set(lot_split(small_df, random_state=999)[2]["lot_id"])
        # With 12 lots and different seeds it is extremely unlikely test sets match
        assert lots_a != lots_b

    def test_raises_without_lot_id_column(self):
        df = pd.DataFrame({"temperature": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="lot_id"):
            lot_split(df)

    def test_raises_on_fractions_not_summing_to_one(self, small_df):
        with pytest.raises(ValueError, match="sum to 1.0"):
            lot_split(small_df, train_frac=0.5, val_frac=0.3, test_frac=0.3)

    def test_raises_on_too_few_lots(self):
        df = pd.DataFrame({"lot_id": ["A", "B"], "x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Not enough lots"):
            lot_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15)

    def test_reset_index_in_each_split(self, split_dfs):
        for df in split_dfs:
            assert list(df.index) == list(range(len(df)))


# ── Feature set definitions ────────────────────────────────────────────────────

class TestFeatureSetDefinitions:
    # process_only: in-situ only (no offline metrology)
    def test_process_only_excludes_film_thickness(self):
        assert "film_thickness" not in PROCESS_ONLY_FEATURES

    def test_process_only_excludes_overlay_error(self):
        assert "overlay_error" not in PROCESS_ONLY_FEATURES

    def test_process_only_excludes_defect_density(self):
        assert "defect_density" not in PROCESS_ONLY_FEATURES

    def test_process_only_includes_sensor_features(self):
        assert all(f in PROCESS_ONLY_FEATURES for f in SENSOR_FEATURES)

    def test_process_only_includes_exposure_dose(self):
        assert "exposure_dose" in PROCESS_ONLY_FEATURES

    def test_process_only_excludes_yield_rate(self):
        assert not any(f in PROCESS_ONLY_FEATURES for f in YIELD_FEATURES)

    def test_process_only_excludes_label_cols(self):
        assert not any(f in PROCESS_ONLY_FEATURES for f in LABEL_COLS)

    # process_plus_metrology: adds offline inspection results
    def test_process_plus_metrology_includes_offline_features(self):
        for f in ("film_thickness", "overlay_error", "defect_density"):
            assert f in PROCESS_PLUS_METROLOGY_FEATURES

    def test_process_plus_metrology_excludes_yield_rate(self):
        assert not any(f in PROCESS_PLUS_METROLOGY_FEATURES for f in YIELD_FEATURES)

    def test_process_plus_metrology_excludes_label_cols(self):
        assert not any(f in PROCESS_PLUS_METROLOGY_FEATURES for f in LABEL_COLS)

    # subset / alias relationships
    def test_process_only_is_subset_of_full(self):
        assert set(PROCESS_ONLY_FEATURES).issubset(set(PROCESS_PLUS_METROLOGY_FEATURES))

    def test_anomaly_input_features_is_alias_for_full(self):
        assert ANOMALY_INPUT_FEATURES == PROCESS_PLUS_METROLOGY_FEATURES

    def test_feature_set_map_keys(self):
        assert set(FEATURE_SET_MAP.keys()) == {"process_only", "full"}

    def test_feature_set_map_process_only_value(self):
        assert FEATURE_SET_MAP["process_only"] is PROCESS_ONLY_FEATURES

    def test_feature_set_map_full_value(self):
        assert FEATURE_SET_MAP["full"] is PROCESS_PLUS_METROLOGY_FEATURES


# ── features.py ───────────────────────────────────────────────────────────────

class TestFeatureDefinitions:
    def test_anomaly_input_features_excludes_yield_rate(self):
        assert not any(f in ANOMALY_INPUT_FEATURES for f in YIELD_FEATURES)

    def test_anomaly_input_features_excludes_label_cols(self):
        assert not any(f in ANOMALY_INPUT_FEATURES for f in LABEL_COLS)

    def test_anomaly_input_features_contains_sensor_features(self):
        from semiconductor_yield.process.features import SENSOR_FEATURES
        assert all(f in ANOMALY_INPUT_FEATURES for f in SENSOR_FEATURES)

    def test_anomaly_input_features_contains_metrology_features(self):
        from semiconductor_yield.process.features import METROLOGY_FEATURES
        assert all(f in ANOMALY_INPUT_FEATURES for f in METROLOGY_FEATURES)

    def test_label_cols_not_in_anomaly_features(self):
        for col in LABEL_COLS:
            assert col not in ANOMALY_INPUT_FEATURES


class TestGetFeatureMatrix:
    def test_returns_tuple_of_array_and_list(self, small_df):
        result = get_feature_matrix(small_df)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], list)

    def test_output_shape_matches_input(self, small_df):
        X, cols = get_feature_matrix(small_df)
        assert X.shape[0] == len(small_df)
        assert X.shape[1] == len(cols)

    def test_no_nan_in_output(self, small_df):
        X, _ = get_feature_matrix(small_df)
        assert not np.isnan(X).any()

    def test_dtype_is_float64(self, small_df):
        X, _ = get_feature_matrix(small_df)
        assert X.dtype == np.float64

    def test_yield_rate_not_in_output_by_default(self, small_df):
        _, cols = get_feature_matrix(small_df)
        assert "yield_rate" not in cols

    def test_anomaly_label_not_in_output_by_default(self, small_df):
        _, cols = get_feature_matrix(small_df)
        assert "anomaly_label" not in cols

    def test_raises_if_no_columns_present(self):
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="None of the requested"):
            get_feature_matrix(df, feature_cols=["temperature"])

    def test_skips_absent_columns(self, small_df):
        X, cols = get_feature_matrix(small_df, feature_cols=["temperature", "__nonexistent__"])
        assert "__nonexistent__" not in cols
        assert "temperature" in cols

    def test_all_nan_column_imputed_to_zero(self):
        df = pd.DataFrame({
            "a": [float("nan"), float("nan"), float("nan")],
            "b": [1.0, 2.0, 3.0],
        })
        X, _ = get_feature_matrix(df, feature_cols=["a", "b"])
        assert np.all(X[:, 0] == 0.0)


# ── IsolationForestDetector ────────────────────────────────────────────────────

class TestIsolationForestDetector:
    def test_fit_returns_self(self, feature_matrix):
        X, cols = feature_matrix
        det = IsolationForestDetector(n_estimators=10, random_state=0)
        assert det.fit(X, cols) is det

    def test_anomaly_scores_length(self, fitted_if, feature_matrix):
        X, _ = feature_matrix
        scores = fitted_if.anomaly_scores(X)
        assert len(scores) == len(X)

    def test_anomaly_scores_are_finite(self, fitted_if, feature_matrix):
        X, _ = feature_matrix
        assert np.all(np.isfinite(fitted_if.anomaly_scores(X)))

    def test_predict_returns_binary(self, fitted_if, feature_matrix):
        X, _ = feature_matrix
        preds = fitted_if.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_length_equals_input(self, fitted_if, feature_matrix):
        X, _ = feature_matrix
        assert len(fitted_if.predict(X)) == len(X)

    def test_score_and_predict_returns_anomaly_result(self, fitted_if, feature_matrix):
        from semiconductor_yield.process.anomaly import AnomalyResult
        X, _ = feature_matrix
        result = fitted_if.score_and_predict(X)
        assert isinstance(result, AnomalyResult)
        assert len(result.scores) == len(X)
        assert len(result.labels_pred) == len(X)

    def test_threshold_is_positive_finite(self, fitted_if):
        assert np.isfinite(fitted_if._threshold)

    def test_feature_importances_shape(self, fitted_if, feature_matrix):
        _, cols = feature_matrix
        imp = fitted_if.feature_importances()
        assert len(imp) == len(cols)

    def test_feature_importances_sum_to_one(self, fitted_if):
        imp = fitted_if.feature_importances()
        assert imp.sum() == pytest.approx(1.0, abs=0.01)

    def test_save_and_load_roundtrip(self, fitted_if, feature_matrix, tmp_path):
        X, _ = feature_matrix
        path = tmp_path / "if_model.joblib"
        fitted_if.save(path)
        assert path.exists()

        loaded = IsolationForestDetector.load(path)
        np.testing.assert_allclose(
            fitted_if.anomaly_scores(X), loaded.anomaly_scores(X), rtol=1e-6
        )

    def test_save_preserves_threshold(self, fitted_if, tmp_path):
        path = tmp_path / "if_threshold.joblib"
        fitted_if.save(path)
        loaded = IsolationForestDetector.load(path)
        assert loaded._threshold == pytest.approx(fitted_if._threshold)

    def test_save_preserves_feature_cols(self, fitted_if, tmp_path):
        path = tmp_path / "if_feats.joblib"
        fitted_if.save(path)
        loaded = IsolationForestDetector.load(path)
        assert loaded._feature_cols == fitted_if._feature_cols

    def test_injects_clear_anomaly_gets_high_score(self, feature_matrix):
        X, cols = feature_matrix
        det = IsolationForestDetector(n_estimators=20, random_state=0)
        det.fit(X, cols)
        # Create extreme outlier: all features set to 100× training std
        X_extreme = X.copy()
        X_extreme[0] = np.nanmean(X, axis=0) + 100 * np.nanstd(X, axis=0)
        scores = det.anomaly_scores(X_extreme)
        # The injected outlier should rank highest among all scores
        assert scores[0] == scores.max()


# ── AutoencoderDetector ────────────────────────────────────────────────────────

class TestAutoencoderDetector:
    def test_fit_returns_self(self, feature_matrix):
        X, cols = feature_matrix
        det = AutoencoderDetector(hidden_dims=(4, 2), epochs=2, batch_size=16)
        assert det.fit(X, cols) is det

    def test_anomaly_scores_length(self, fitted_ae, feature_matrix):
        X, _ = feature_matrix
        assert len(fitted_ae.anomaly_scores(X)) == len(X)

    def test_anomaly_scores_nonnegative(self, fitted_ae, feature_matrix):
        X, _ = feature_matrix
        assert np.all(fitted_ae.anomaly_scores(X) >= 0)

    def test_predict_returns_binary(self, fitted_ae, feature_matrix):
        X, _ = feature_matrix
        preds = fitted_ae.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_length_equals_input(self, fitted_ae, feature_matrix):
        X, _ = feature_matrix
        assert len(fitted_ae.predict(X)) == len(X)

    def test_score_and_predict_returns_anomaly_result(self, fitted_ae, feature_matrix):
        from semiconductor_yield.process.anomaly import AnomalyResult
        X, _ = feature_matrix
        result = fitted_ae.score_and_predict(X)
        assert isinstance(result, AnomalyResult)

    def test_per_feature_reconstruction_error_shape(self, fitted_ae, feature_matrix):
        X, cols = feature_matrix
        err = fitted_ae.per_feature_reconstruction_error(X)
        assert len(err) == len(cols)

    def test_per_feature_reconstruction_error_nonnegative(self, fitted_ae, feature_matrix):
        X, _ = feature_matrix
        err = fitted_ae.per_feature_reconstruction_error(X)
        assert np.all(err >= 0)

    def test_save_and_load_roundtrip(self, fitted_ae, feature_matrix, tmp_path):
        X, _ = feature_matrix
        path = tmp_path / "ae_model.joblib"
        fitted_ae.save(path)
        assert path.exists()

        loaded = AutoencoderDetector.load(path)
        np.testing.assert_allclose(
            fitted_ae.anomaly_scores(X), loaded.anomaly_scores(X), rtol=1e-5
        )

    def test_save_preserves_threshold(self, fitted_ae, tmp_path):
        path = tmp_path / "ae_threshold.joblib"
        fitted_ae.save(path)
        loaded = AutoencoderDetector.load(path)
        assert loaded._threshold == pytest.approx(fitted_ae._threshold)

    def test_save_is_single_file(self, fitted_ae, tmp_path):
        path = tmp_path / "ae_single.joblib"
        fitted_ae.save(path)
        # Only one file should exist (no separate .pt sidecar)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "ae_single.joblib"

    def test_threshold_is_positive(self, fitted_ae):
        assert fitted_ae._threshold > 0


# ── Cross-detector ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("DetectorClass,kwargs", [
    (IsolationForestDetector, {"n_estimators": 10}),
    (AutoencoderDetector,     {"hidden_dims": (4, 2), "epochs": 2}),
])
def test_score_length_matches_input(DetectorClass, kwargs, feature_matrix):
    X, cols = feature_matrix
    det = DetectorClass(**kwargs)
    det.fit(X, cols)
    assert len(det.anomaly_scores(X)) == len(X)


@pytest.mark.parametrize("DetectorClass,kwargs", [
    (IsolationForestDetector, {"n_estimators": 10}),
    (AutoencoderDetector,     {"hidden_dims": (4, 2), "epochs": 2}),
])
def test_predict_on_new_data(DetectorClass, kwargs, feature_matrix):
    X, cols = feature_matrix
    det = DetectorClass(**kwargs)
    det.fit(X, cols)
    X_new = X[:10]
    preds = det.predict(X_new)
    assert len(preds) == 10
    assert set(np.unique(preds)).issubset({0, 1})
