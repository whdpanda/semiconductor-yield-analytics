"""Smoke tests — verify the package is importable and core paths resolve correctly.

These tests must pass on a fresh clone with no data downloaded.
"""

import importlib
from pathlib import Path


def test_package_importable():
    mod = importlib.import_module("semiconductor_yield")
    assert mod.__version__ == "0.1.0"


def test_version_string_format():
    import semiconductor_yield

    parts = semiconductor_yield.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_config_root_dir_exists():
    from semiconductor_yield.config import ROOT_DIR

    assert ROOT_DIR.exists(), f"ROOT_DIR not found: {ROOT_DIR}"


def test_config_data_dir_structure():
    from semiconductor_yield.config import DATA_DIR, RAW_DIR, SYNTHETIC_DIR

    # These directories should exist (created during project setup)
    for d in [DATA_DIR, RAW_DIR, SYNTHETIC_DIR]:
        assert d.exists(), f"Expected directory missing: {d}"


def test_config_wafer_classes():
    from semiconductor_yield.config import WAFER_DEFECT_CLASSES, NUM_WAFER_CLASSES

    assert NUM_WAFER_CLASSES == 9
    assert "none" in WAFER_DEFECT_CLASSES
    assert "Center" in WAFER_DEFECT_CLASSES


def test_config_random_seed_type():
    from semiconductor_yield.config import RANDOM_SEED

    assert isinstance(RANDOM_SEED, int)


def test_ensure_output_dirs_creates_dirs(tmp_path, monkeypatch):
    """ensure_output_dirs() must create missing directories without error."""
    import semiconductor_yield.config as cfg

    # Redirect outputs to a temp directory so we don't touch the real project
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(cfg, "GRADCAM_DIR", tmp_path / "reports" / "gradcam")
    monkeypatch.setattr(cfg, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(cfg, "WAFER_MAPS_DIR", tmp_path / "processed" / "wafer_maps")
    monkeypatch.setattr(cfg, "SPC_PROCESSED_DIR", tmp_path / "processed" / "spc")

    cfg.ensure_output_dirs()

    assert (tmp_path / "models").exists()
    assert (tmp_path / "reports").exists()
    assert (tmp_path / "logs").exists()


def test_submodules_importable():
    modules = [
        "semiconductor_yield.data",
        "semiconductor_yield.wafer",
        "semiconductor_yield.process",
        "semiconductor_yield.models",
        "semiconductor_yield.dashboard",
        "semiconductor_yield.utils",
    ]
    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None, f"Failed to import {name}"
