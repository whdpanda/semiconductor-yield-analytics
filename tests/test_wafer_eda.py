"""Tests for WM-811K wafer map EDA utilities (Module A).

All tests use synthetic wafer maps — the real LSWMD.pkl is not required.
matplotlib is set to the Agg backend via tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from semiconductor_yield.config import WAFER_DEFECT_CLASSES
from semiconductor_yield.wafer.data_loader import LABEL_TO_IDX, WaferSample
from semiconductor_yield.wafer.eda import (
    WAFER_REPORTS_DIR,
    class_distribution,
    plot_class_distribution,
    plot_sample_wafer_maps,
    run_eda,
    wafer_map_size_distribution,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_wmap(h: int = 16, w: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 3, size=(h, w)).astype(np.float32)


def _make_samples(n_per_class: int = 5, seed: int = 0) -> list[WaferSample]:
    rng = np.random.default_rng(seed)
    samples: list[WaferSample] = []
    idx = 0
    for class_name in WAFER_DEFECT_CLASSES:
        for k in range(n_per_class):
            samples.append(
                WaferSample(
                    wafer_map=rng.integers(0, 3, size=(16, 16)).astype(np.float32),
                    label=LABEL_TO_IDX[class_name],
                    label_name=class_name,
                    lot_name=f"LOT_{idx // 5:04d}",
                    wafer_index=idx,
                    split="train",
                )
            )
            idx += 1
    return samples


def _make_raw_df(n: int = 20) -> pd.DataFrame:
    """Minimal fake raw DataFrame with waferMap column."""
    rng = np.random.default_rng(42)
    rows = []
    sizes = [(26, 26)] * 14 + [(52, 52)] * 4 + [(17, 17)] * 2
    for i, (h, w) in enumerate(sizes[:n]):
        rows.append({"waferMap": rng.integers(0, 3, size=(h, w)).astype(np.uint8)})
    return pd.DataFrame(rows)


# ── class_distribution ─────────────────────────────────────────────────────────


def test_class_distribution_returns_series():
    assert isinstance(class_distribution(_make_samples(3)), pd.Series)


def test_class_distribution_has_all_9_classes():
    dist = class_distribution(_make_samples(3))
    assert set(dist.index) == set(WAFER_DEFECT_CLASSES)


def test_class_distribution_counts_correct():
    samples = _make_samples(n_per_class=4)
    dist = class_distribution(samples)
    for class_name in WAFER_DEFECT_CLASSES:
        assert dist[class_name] == 4


def test_class_distribution_sorted_descending():
    samples = _make_samples(n_per_class=5)
    dist = class_distribution(samples)
    assert (dist.diff().dropna() <= 0).all()


def test_class_distribution_empty_class_gets_zero():
    # All samples belong to "none"; other classes should be 0
    samples = [
        WaferSample(
            wafer_map=_make_wmap(),
            label=LABEL_TO_IDX["none"],
            label_name="none",
            lot_name="L001",
            wafer_index=i,
            split="train",
        )
        for i in range(5)
    ]
    dist = class_distribution(samples)
    assert dist["Center"] == 0
    assert dist["none"] == 5


def test_class_distribution_total_matches_input():
    samples = _make_samples(n_per_class=7)
    dist = class_distribution(samples)
    assert dist.sum() == len(samples)


# ── wafer_map_size_distribution ────────────────────────────────────────────────


def test_size_distribution_returns_dataframe():
    assert isinstance(wafer_map_size_distribution(_make_raw_df()), pd.DataFrame)


def test_size_distribution_has_required_columns():
    df = wafer_map_size_distribution(_make_raw_df())
    assert {"height", "width", "count", "percentage"}.issubset(set(df.columns))


def test_size_distribution_count_sums_to_total():
    raw = _make_raw_df(20)
    df = wafer_map_size_distribution(raw)
    assert df["count"].sum() == len(raw)


def test_size_distribution_percentage_sums_to_100():
    raw = _make_raw_df(20)
    df = wafer_map_size_distribution(raw)
    assert df["percentage"].sum() == pytest.approx(100.0, abs=0.01)


def test_size_distribution_sorted_descending():
    df = wafer_map_size_distribution(_make_raw_df(20))
    assert (df["count"].diff().dropna() <= 0).all()


# ── plot_class_distribution ────────────────────────────────────────────────────


def test_plot_class_distribution_creates_file(tmp_path):
    samples = _make_samples(n_per_class=3)
    dist = class_distribution(samples)
    out = tmp_path / "test_dist.png"
    plot_class_distribution(dist, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_class_distribution_creates_parent_dir(tmp_path):
    dist = class_distribution(_make_samples(3))
    out = tmp_path / "subdir" / "nested" / "dist.png"
    plot_class_distribution(dist, out)
    assert out.exists()


# ── plot_sample_wafer_maps ─────────────────────────────────────────────────────


def test_plot_sample_wafer_maps_creates_file(tmp_path):
    samples = _make_samples(n_per_class=3)
    out = tmp_path / "wafer_maps.png"
    plot_sample_wafer_maps(samples, out, n_per_class=2)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_sample_wafer_maps_with_sparse_classes(tmp_path):
    # Only 'none' class has samples — other classes show placeholder cells
    samples = [
        WaferSample(
            wafer_map=_make_wmap(seed=i),
            label=LABEL_TO_IDX["none"],
            label_name="none",
            lot_name="L001",
            wafer_index=i,
            split="train",
        )
        for i in range(3)
    ]
    out = tmp_path / "sparse.png"
    plot_sample_wafer_maps(samples, out, n_per_class=2)
    assert out.exists()


# ── run_eda (integration) ──────────────────────────────────────────────────────


def test_run_eda_returns_dict(tmp_path):
    outputs = run_eda(_make_samples(n_per_class=3), output_dir=tmp_path)
    assert isinstance(outputs, dict)


def test_run_eda_creates_required_files(tmp_path):
    outputs = run_eda(_make_samples(n_per_class=3), output_dir=tmp_path)
    required = {
        "class_distribution_csv",
        "class_distribution_png",
        "imbalance_report",
        "sample_wafer_maps_png",
    }
    for key in required:
        assert key in outputs, f"Missing output key: {key}"
        assert outputs[key].exists(), f"File not created: {outputs[key]}"


def test_run_eda_class_distribution_csv_content(tmp_path):
    samples = _make_samples(n_per_class=4)
    outputs = run_eda(samples, output_dir=tmp_path)
    df = pd.read_csv(outputs["class_distribution_csv"])
    assert "class_name" in df.columns
    assert "count" in df.columns
    assert df["count"].sum() == len(samples)


def test_run_eda_with_raw_df_creates_size_csv(tmp_path):
    samples = _make_samples(n_per_class=3)
    raw_df = _make_raw_df(20)
    outputs = run_eda(samples, df_raw=raw_df, output_dir=tmp_path)
    assert "size_distribution_csv" in outputs
    assert outputs["size_distribution_csv"].exists()


def test_run_eda_without_raw_df_no_size_csv(tmp_path):
    outputs = run_eda(_make_samples(n_per_class=3), df_raw=None, output_dir=tmp_path)
    assert "size_distribution_csv" not in outputs


def test_run_eda_creates_output_dir_if_missing(tmp_path):
    new_dir = tmp_path / "brand_new_dir"
    assert not new_dir.exists()
    run_eda(_make_samples(n_per_class=3), output_dir=new_dir)
    assert new_dir.exists()


# ── WAFER_REPORTS_DIR constant ─────────────────────────────────────────────────


def test_wafer_reports_dir_is_path():
    from pathlib import Path
    assert isinstance(WAFER_REPORTS_DIR, Path)


def test_wafer_reports_dir_under_reports():
    from semiconductor_yield.config import REPORTS_DIR
    assert WAFER_REPORTS_DIR.parent == REPORTS_DIR
