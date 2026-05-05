"""Tests for wafer map preprocessing utilities (Module A).

All tests use synthetic wafer maps — the real LSWMD.pkl is not required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from semiconductor_yield.config import WAFER_DEFECT_CLASSES, WAFER_MAP_SIZE
from semiconductor_yield.wafer.data_loader import LABEL_TO_IDX, WaferSample
from semiconductor_yield.wafer.preprocess import (
    SplitResult,
    compute_class_weights,
    decode_label,
    encode_label,
    imbalance_stats,
    normalize_wafer_map,
    resize_wafer_map,
    stratified_split,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

_CLASSES = WAFER_DEFECT_CLASSES  # 9 classes


def _make_wafer_map(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=(h, w)).astype(np.float32)


def _make_samples(n_per_class: int = 10, seed: int = 0) -> list[WaferSample]:
    """Create n_per_class samples for each of the 9 defect classes (90 total)."""
    rng = np.random.default_rng(seed)
    samples: list[WaferSample] = []
    idx = 0
    for class_name in _CLASSES:
        for k in range(n_per_class):
            wmap = rng.integers(0, 3, size=(32, 32)).astype(np.float32)
            samples.append(
                WaferSample(
                    wafer_map=wmap,
                    label=LABEL_TO_IDX[class_name],
                    label_name=class_name,
                    lot_name=f"LOT_{idx // 5:04d}",
                    wafer_index=idx,
                    split="train",
                )
            )
            idx += 1
    return samples


# ── resize_wafer_map ───────────────────────────────────────────────────────────


def test_resize_output_shape():
    wmap = _make_wafer_map(20, 30)
    result = resize_wafer_map(wmap, target_size=(64, 64))
    assert result.shape == (64, 64)


def test_resize_default_target_size():
    wmap = _make_wafer_map(20, 30)
    result = resize_wafer_map(wmap)
    assert result.shape == WAFER_MAP_SIZE


def test_resize_preserves_discrete_values():
    wmap = _make_wafer_map(16, 16)
    result = resize_wafer_map(wmap, target_size=(64, 64))
    unique_vals = set(np.unique(result))
    assert unique_vals <= {0.0, 1.0, 2.0}, f"Unexpected values after resize: {unique_vals}"


def test_resize_output_dtype():
    result = resize_wafer_map(_make_wafer_map())
    assert result.dtype == np.float32


def test_resize_no_value_blending():
    # All-zeros map should stay all-zeros after resize
    wmap = np.zeros((10, 10), dtype=np.float32)
    result = resize_wafer_map(wmap, target_size=(40, 40))
    assert np.all(result == 0.0)


def test_resize_identity_when_already_correct_size():
    wmap = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.float32)
    result = resize_wafer_map(wmap, target_size=(2, 3))
    np.testing.assert_array_equal(result, wmap)


# ── normalize_wafer_map ────────────────────────────────────────────────────────


def test_normalize_maps_0_to_0():
    assert normalize_wafer_map(np.array([[0.0]])).item() == 0.0


def test_normalize_maps_1_to_half():
    assert normalize_wafer_map(np.array([[1.0]])).item() == pytest.approx(0.5)


def test_normalize_maps_2_to_1():
    assert normalize_wafer_map(np.array([[2.0]])).item() == pytest.approx(1.0)


def test_normalize_output_dtype():
    result = normalize_wafer_map(np.array([[0, 1, 2]], dtype=np.uint8))
    assert result.dtype == np.float32


def test_normalize_range():
    wmap = _make_wafer_map()
    result = normalize_wafer_map(wmap)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


# ── encode_label / decode_label ────────────────────────────────────────────────


def test_encode_label_known_class():
    assert encode_label("Center") == LABEL_TO_IDX["Center"]


def test_encode_label_all_classes():
    for class_name in WAFER_DEFECT_CLASSES:
        idx = encode_label(class_name)
        assert isinstance(idx, int)
        assert 0 <= idx < len(WAFER_DEFECT_CLASSES)


def test_encode_label_unknown_raises():
    with pytest.raises(KeyError, match="Unknown class"):
        encode_label("not_a_real_class")


def test_decode_label_roundtrip():
    for class_name in WAFER_DEFECT_CLASSES:
        idx = encode_label(class_name)
        assert decode_label(idx) == class_name


def test_decode_label_unknown_raises():
    with pytest.raises(KeyError, match="Unknown index"):
        decode_label(999)


# ── stratified_split ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def balanced_samples() -> list[WaferSample]:
    return _make_samples(n_per_class=10)  # 90 total, 10 per class


def test_split_total_preserved(balanced_samples):
    result = stratified_split(balanced_samples)
    assert result.total == len(balanced_samples)


def test_split_train_ratio_approximate(balanced_samples):
    n = len(balanced_samples)
    result = stratified_split(balanced_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    assert abs(len(result.train) / n - 0.70) < 0.06


def test_split_val_ratio_approximate(balanced_samples):
    n = len(balanced_samples)
    result = stratified_split(balanced_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    assert abs(len(result.val) / n - 0.15) < 0.06


def test_split_test_ratio_approximate(balanced_samples):
    n = len(balanced_samples)
    result = stratified_split(balanced_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    assert abs(len(result.test) / n - 0.15) < 0.06


def test_split_no_overlap_train_val(balanced_samples):
    result = stratified_split(balanced_samples)
    train_ids = {id(s) for s in result.train}
    val_ids = {id(s) for s in result.val}
    assert not (train_ids & val_ids), "Same object appears in both train and val"


def test_split_no_overlap_train_test(balanced_samples):
    result = stratified_split(balanced_samples)
    train_ids = {id(s) for s in result.train}
    test_ids = {id(s) for s in result.test}
    assert not (train_ids & test_ids), "Same object appears in both train and test"


def test_split_no_overlap_val_test(balanced_samples):
    result = stratified_split(balanced_samples)
    val_ids = {id(s) for s in result.val}
    test_ids = {id(s) for s in result.test}
    assert not (val_ids & test_ids), "Same object appears in both val and test"


def test_split_covers_all_samples(balanced_samples):
    result = stratified_split(balanced_samples)
    all_ids = (
        {id(s) for s in result.train}
        | {id(s) for s in result.val}
        | {id(s) for s in result.test}
    )
    original_ids = {id(s) for s in balanced_samples}
    assert all_ids == original_ids, "Some samples not assigned to any split"


def test_split_reproducible(balanced_samples):
    r1 = stratified_split(balanced_samples, seed=7)
    r2 = stratified_split(balanced_samples, seed=7)
    assert [id(s) for s in r1.train] == [id(s) for s in r2.train]
    assert [id(s) for s in r1.val] == [id(s) for s in r2.val]
    assert [id(s) for s in r1.test] == [id(s) for s in r2.test]


def test_split_different_seeds_differ(balanced_samples):
    r1 = stratified_split(balanced_samples, seed=1)
    r2 = stratified_split(balanced_samples, seed=2)
    assert [id(s) for s in r1.train] != [id(s) for s in r2.train]


def test_split_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        stratified_split([])


def test_split_invalid_ratios(balanced_samples):
    with pytest.raises(ValueError, match="sum to 1.0"):
        stratified_split(balanced_samples, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


def test_split_result_sizes_method(balanced_samples):
    result = stratified_split(balanced_samples)
    sizes = result.sizes()
    assert set(sizes.keys()) == {"train", "val", "test"}
    assert all(isinstance(v, int) for v in sizes.values())


def test_split_all_classes_in_train(balanced_samples):
    result = stratified_split(balanced_samples)
    train_classes = {s.label for s in result.train}
    assert len(train_classes) == len(WAFER_DEFECT_CLASSES), (
        "Not all classes represented in train split"
    )


# ── compute_class_weights ──────────────────────────────────────────────────────


def test_class_weights_shape():
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8] * 10
    weights = compute_class_weights(labels, n_classes=9)
    assert weights.shape == (9,)


def test_class_weights_balanced_classes_equal():
    # With exactly equal class counts, all weights should be equal.
    labels = list(range(9)) * 20  # 20 per class
    weights = compute_class_weights(labels, n_classes=9)
    assert np.allclose(weights, weights[0]), "Equal classes should have equal weights"


def test_class_weights_rare_class_gets_higher_weight():
    # Class 0 has 1 sample; class 1 has 9 samples → weight[0] > weight[1]
    labels = [0] + [1] * 9
    weights = compute_class_weights(labels, n_classes=2)
    assert weights[0] > weights[1]


def test_class_weights_missing_class_gets_zero():
    # Labels contain only classes 0 and 1; class 2 is absent.
    labels = [0, 0, 1, 1]
    weights = compute_class_weights(labels, n_classes=3)
    assert weights[2] == 0.0


def test_class_weights_dtype():
    labels = list(range(9)) * 5
    weights = compute_class_weights(labels)
    assert weights.dtype == np.float64


# ── imbalance_stats ────────────────────────────────────────────────────────────


def test_imbalance_stats_returns_dataframe():
    assert isinstance(imbalance_stats(_make_samples(5)), pd.DataFrame)


def test_imbalance_stats_has_all_columns():
    expected = {"class_name", "count", "percentage", "imbalance_ratio", "weight"}
    result = imbalance_stats(_make_samples(5))
    assert expected.issubset(set(result.columns))


def test_imbalance_stats_count_sum_equals_total():
    samples = _make_samples(n_per_class=7)
    stats = imbalance_stats(samples)
    assert stats["count"].sum() == len(samples)


def test_imbalance_stats_sorted_descending():
    samples = _make_samples(n_per_class=5)
    stats = imbalance_stats(samples)
    assert (stats["count"].diff().dropna() <= 0).all()


def test_imbalance_stats_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        imbalance_stats([])


def test_imbalance_stats_majority_class_ratio_is_one():
    samples = _make_samples(n_per_class=5)
    stats = imbalance_stats(samples)
    # The majority class (first row after sort) must have ratio = 1.0
    assert stats.iloc[0]["imbalance_ratio"] == pytest.approx(1.0)


def test_imbalance_stats_percentages_sum_to_100():
    samples = _make_samples(n_per_class=5)
    stats = imbalance_stats(samples)
    # Rounding each class to 2 d.p. accumulates up to ~0.05 across 9 classes.
    assert stats["percentage"].sum() == pytest.approx(100.0, abs=0.1)
