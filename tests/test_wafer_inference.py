"""Tests for wafer inference pipeline and synthetic demo sample generators.

No WM-811K data or trained checkpoint required.
All inference tests run in WaferInference.demo() mode (random weights).
"""

from __future__ import annotations

import io
import pickle

import numpy as np
import pytest

from semiconductor_yield.config import WAFER_DEFECT_CLASSES
from semiconductor_yield.wafer.demo_samples import (
    DEMO_GENERATORS,
    generate_demo_sample,
    make_center_pattern,
    make_donut_pattern,
    make_edge_loc_pattern,
    make_edge_ring_pattern,
    make_loc_pattern,
    make_near_full_pattern,
    make_none_pattern,
    make_random_pattern,
    make_scratch_pattern,
)
from semiconductor_yield.wafer.inference import (
    InferenceResult,
    WaferInference,
    parse_wafer_input,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _random_raw_map(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """Synthetic wafer map with discrete {0, 1, 2} values."""
    rng = np.random.default_rng(seed)
    wmap = np.ones((h, w), dtype=np.float32)
    mask = rng.random((h, w)) < 0.1
    wmap[mask] = 2.0
    return wmap


# ── TestDemoSamples ────────────────────────────────────────────────────────────

class TestDemoSamples:
    def test_registry_covers_all_classes(self):
        assert set(DEMO_GENERATORS.keys()) == set(WAFER_DEFECT_CLASSES)

    @pytest.mark.parametrize("cls", list(WAFER_DEFECT_CLASSES))
    def test_generate_demo_sample_shape(self, cls):
        arr = generate_demo_sample(cls)
        assert arr.shape == (64, 64), f"{cls}: expected (64, 64), got {arr.shape}"

    @pytest.mark.parametrize("cls", list(WAFER_DEFECT_CLASSES))
    def test_generate_demo_sample_dtype(self, cls):
        arr = generate_demo_sample(cls)
        assert arr.dtype == np.float32, f"{cls}: expected float32"

    @pytest.mark.parametrize("cls", list(WAFER_DEFECT_CLASSES))
    def test_values_are_valid(self, cls):
        arr = generate_demo_sample(cls)
        unique = set(np.unique(arr).tolist())
        assert unique <= {0.0, 1.0, 2.0}, f"{cls}: unexpected values {unique}"

    def test_corners_are_background(self):
        """Wafer is circular — corners of the square grid must be background (0)."""
        for cls in WAFER_DEFECT_CLASSES:
            arr = generate_demo_sample(cls, seed=0)
            corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
            assert all(v == 0.0 for v in corners), (
                f"{cls}: corner pixel should be background (0), got {corners}"
            )

    def test_defect_patterns_have_defects(self):
        """All classes except 'none' should have at least one defective die."""
        for cls in WAFER_DEFECT_CLASSES:
            if cls == "none":
                continue
            arr = generate_demo_sample(cls, seed=42)
            assert (arr == 2.0).any(), f"{cls}: no defective dies found in pattern"

    def test_none_pattern_low_defect_rate(self):
        """'none' class should have < 5% defect rate inside the wafer boundary."""
        arr = generate_demo_sample("none", seed=42)
        in_wafer = arr > 0
        defective = arr == 2.0
        if in_wafer.sum() > 0:
            rate = defective.sum() / in_wafer.sum()
            assert rate < 0.05, f"'none' defect rate {rate:.2%} >= 5%"

    def test_seed_reproducibility(self):
        arr1 = generate_demo_sample("Center", seed=7)
        arr2 = generate_demo_sample("Center", seed=7)
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_seeds_produce_different_maps(self):
        arr1 = generate_demo_sample("Random", seed=1)
        arr2 = generate_demo_sample("Random", seed=999)
        assert not np.array_equal(arr1, arr2), (
            "Different seeds should produce different wafer maps for 'Random'"
        )

    def test_invalid_class_raises(self):
        with pytest.raises(KeyError, match="Unknown class"):
            generate_demo_sample("NotAClass")

    @pytest.mark.parametrize("fn,cls", [
        (make_center_pattern,    "Center"),
        (make_donut_pattern,     "Donut"),
        (make_edge_ring_pattern, "Edge-Ring"),
        (make_edge_loc_pattern,  "Edge-Loc"),
        (make_loc_pattern,       "Loc"),
        (make_near_full_pattern, "Near-full"),
        (make_random_pattern,    "Random"),
        (make_scratch_pattern,   "Scratch"),
        (make_none_pattern,      "none"),
    ])
    def test_generator_callable_directly(self, fn, cls):
        arr = fn(size=64, seed=0)
        assert arr.shape == (64, 64)
        assert arr.dtype == np.float32


# ── TestInferenceResult ────────────────────────────────────────────────────────

class TestInferenceResult:
    def _result(self, top_k: int = 5) -> InferenceResult:
        engine = WaferInference.demo()
        return engine.predict(_random_raw_map(), top_k=top_k)

    def test_fields_present(self):
        r = self._result()
        for field in ("predicted_class", "class_index", "confidence", "top_k",
                      "preprocessed_map", "is_demo"):
            assert hasattr(r, field), f"Missing field: {field}"

    def test_predicted_class_in_vocab(self):
        r = self._result()
        assert r.predicted_class in WAFER_DEFECT_CLASSES

    def test_class_index_valid(self):
        r = self._result()
        assert 0 <= r.class_index < len(WAFER_DEFECT_CLASSES)

    def test_class_index_matches_name(self):
        r = self._result()
        assert list(WAFER_DEFECT_CLASSES)[r.class_index] == r.predicted_class

    def test_confidence_in_range(self):
        r = self._result()
        assert 0.0 <= r.confidence <= 1.0

    def test_top_k_sorted_descending(self):
        r = self._result(top_k=5)
        probs = [p for _, p in r.top_k]
        assert probs == sorted(probs, reverse=True), "top_k not sorted descending"

    def test_confidence_matches_top1_prob(self):
        r = self._result(top_k=9)
        assert abs(r.confidence - r.top_k[0][1]) < 1e-6

    def test_preprocessed_map_shape(self):
        r = self._result()
        assert r.preprocessed_map.shape == (64, 64)

    def test_preprocessed_map_range(self):
        r = self._result()
        assert r.preprocessed_map.min() >= 0.0
        assert r.preprocessed_map.max() <= 1.0

    def test_all_probs_sum_to_one(self):
        engine = WaferInference.demo()
        r = engine.predict(_random_raw_map(), top_k=len(WAFER_DEFECT_CLASSES))
        total = sum(p for _, p in r.top_k)
        assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, not 1.0"

    def test_is_demo_true_in_demo_mode(self):
        r = self._result()
        assert r.is_demo is True


# ── TestWaferInference ─────────────────────────────────────────────────────────

class TestWaferInference:
    def test_demo_creates_without_checkpoint(self):
        engine = WaferInference.demo()
        assert engine is not None
        assert engine._is_demo is True

    def test_from_checkpoint_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            WaferInference.from_checkpoint(tmp_path / "missing.pth")

    def test_predict_returns_inference_result(self):
        engine = WaferInference.demo()
        r = engine.predict(_random_raw_map())
        assert isinstance(r, InferenceResult)

    def test_predict_top_k_length(self):
        engine = WaferInference.demo()
        for k in (1, 3, 5, 9):
            r = engine.predict(_random_raw_map(), top_k=k)
            assert len(r.top_k) == k, f"Expected {k} entries, got {len(r.top_k)}"

    def test_predict_top_k_clamped_to_num_classes(self):
        engine = WaferInference.demo()
        r = engine.predict(_random_raw_map(), top_k=100)
        assert len(r.top_k) == len(WAFER_DEFECT_CLASSES)

    def test_predict_handles_non_standard_size(self):
        """Input with shape != (64, 64) should be resized automatically."""
        engine = WaferInference.demo()
        small = _random_raw_map(h=32, w=32)
        r = engine.predict(small)
        assert r.preprocessed_map.shape == (64, 64)

    def test_predict_batch_length(self):
        engine = WaferInference.demo()
        maps = [_random_raw_map(seed=i) for i in range(4)]
        results = engine.predict_batch(maps)
        assert len(results) == 4

    def test_demo_with_synthetic_pattern(self):
        """Use generate_demo_sample as input to predict()."""
        engine = WaferInference.demo()
        for cls in WAFER_DEFECT_CLASSES:
            wmap = generate_demo_sample(cls)
            r = engine.predict(wmap)
            assert isinstance(r, InferenceResult)


# ── TestParseWaferInput ────────────────────────────────────────────────────────

class TestParseWaferInput:
    def test_passthrough_ndarray(self):
        arr = _random_raw_map()
        result = parse_wafer_input(arr)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_npy_bytes(self):
        arr = _random_raw_map()
        buf = io.BytesIO()
        np.save(buf, arr)
        result = parse_wafer_input(buf.getvalue(), filename="test.npy")
        np.testing.assert_array_almost_equal(result, arr)

    def test_pkl_bytes_from_array(self):
        arr = _random_raw_map()
        data = pickle.dumps(arr)
        result = parse_wafer_input(data, filename="test.pkl")
        np.testing.assert_array_almost_equal(result, arr)

    def test_pkl_bytes_from_dict(self):
        arr = _random_raw_map()
        data = pickle.dumps({"wafer_map": arr, "label": 0})
        result = parse_wafer_input(data, filename="test.pkl")
        np.testing.assert_array_almost_equal(result, arr)

    def test_csv_bytes(self):
        arr = np.array([[0, 1, 2], [1, 1, 2]], dtype=np.float32)
        csv_bytes = "\n".join(",".join(str(v) for v in row) for row in arr).encode()
        result = parse_wafer_input(csv_bytes, filename="test.csv")
        assert result.shape == (2, 3)
        assert result.dtype == np.float32

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            parse_wafer_input(b"data", filename="wafer.xyz")

    def test_wrong_dimensionality_raises(self):
        arr_3d = np.ones((4, 64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            parse_wafer_input(arr_3d)

    def test_pkl_wrong_type_raises(self):
        data = pickle.dumps({"not_wafer_map": [1, 2, 3]})
        with pytest.raises((ValueError, KeyError)):
            parse_wafer_input(data, filename="test.pkl")
