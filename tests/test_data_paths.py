"""Tests for data path configuration and loader error handling.

These tests do NOT require any real datasets to be present.
They verify that paths are correctly configured and that missing-data errors
give actionable messages to the user.
"""

import re
from pathlib import Path

import pytest

from semiconductor_yield import config
from semiconductor_yield.wafer.data_loader import (
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    WM811KLoader,
    _parse_failure_type,
    _parse_split_label,
)


# ── Path configuration ─────────────────────────────────────────────────────────

def test_root_dir_is_path_object():
    assert isinstance(config.ROOT_DIR, Path)


def test_wm811k_pkl_path_is_configured():
    assert isinstance(config.WM811K_PKL, Path)
    assert config.WM811K_PKL.name == "LSWMD.pkl"
    assert "wm811k" in str(config.WM811K_PKL)


def test_secom_dir_is_configured():
    assert isinstance(config.SECOM_DIR, Path)
    assert "secom" in str(config.SECOM_DIR).lower()


def test_synthetic_dir_is_configured():
    assert isinstance(config.SYNTHETIC_DIR, Path)
    assert "synthetic" in str(config.SYNTHETIC_DIR)


def test_wafer_map_size_is_2_tuple():
    h, w = config.WAFER_MAP_SIZE
    assert isinstance(h, int) and h > 0
    assert isinstance(w, int) and w > 0


def test_num_wafer_classes():
    assert config.NUM_WAFER_CLASSES == 9


# ── Label mapping ──────────────────────────────────────────────────────────────

def test_label_to_idx_has_9_classes():
    assert len(LABEL_TO_IDX) == 9


def test_label_to_idx_contains_expected_classes():
    expected = {"Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
                "Near-full", "Random", "Scratch", "none"}
    assert set(LABEL_TO_IDX.keys()) == expected


def test_label_and_idx_maps_are_inverse():
    for name, idx in LABEL_TO_IDX.items():
        assert IDX_TO_LABEL[idx] == name


def test_indices_are_contiguous_from_zero():
    indices = sorted(LABEL_TO_IDX.values())
    assert indices == list(range(9))


# ── _parse_failure_type ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ([["Center"]],   "Center"),
    ([["none"]],     "none"),
    ([["Edge-Loc"]], "Edge-Loc"),
    ([[" Scratch "]],  "Scratch"),    # whitespace should be stripped
    ([[" "]],        None),           # whitespace-only → unlabeled
    ([[""]],         None),           # empty string → unlabeled
    ([[]],           None),           # empty inner list
    ([],             None),           # empty outer list
    (None,           None),
    ("Center",       "Center"),       # plain-string fork of the dataset
    ("",             None),
])
def test_parse_failure_type(raw, expected):
    assert _parse_failure_type(raw) == expected


# ── _parse_split_label ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    (["Training"], "train"),
    (["Test"],     "test"),
    ("Training",   "train"),
    ("Test",       "test"),
    (None,         "train"),
    ("",           "train"),
])
def test_parse_split_label(raw, expected):
    assert _parse_split_label(raw) == expected


# ── WM811KLoader: missing-file error ──────────────────────────────────────────

def test_loader_raises_file_not_found_for_missing_data(tmp_path):
    fake_path = tmp_path / "missing" / "LSWMD.pkl"
    loader = WM811KLoader(pkl_path=fake_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load_raw()
    message = str(exc_info.value)
    # Must point to the correct path
    assert "LSWMD.pkl" in message


def test_loader_error_message_contains_kaggle_url(tmp_path):
    fake_path = tmp_path / "LSWMD.pkl"
    loader = WM811KLoader(pkl_path=fake_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load_raw()
    assert "kaggle.com" in str(exc_info.value).lower()


def test_loader_error_message_contains_target_path(tmp_path):
    fake_path = tmp_path / "LSWMD.pkl"
    loader = WM811KLoader(pkl_path=fake_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load_raw()
    assert str(fake_path) in str(exc_info.value)


def test_loader_default_path_matches_config():
    loader = WM811KLoader()
    assert loader.pkl_path == config.WM811K_PKL


def test_loader_accepts_custom_path(tmp_path):
    custom = tmp_path / "custom.pkl"
    loader = WM811KLoader(pkl_path=custom)
    assert loader.pkl_path == custom


def test_loader_wrong_file_type_raises_type_error(tmp_path):
    """A pkl that doesn't contain a DataFrame should raise TypeError."""
    import pickle

    bad_pkl = tmp_path / "bad.pkl"
    with open(bad_pkl, "wb") as f:
        pickle.dump({"not": "a dataframe"}, f)

    loader = WM811KLoader(pkl_path=bad_pkl)
    with pytest.raises(TypeError, match="DataFrame"):
        loader.load_raw()
