"""Synthetic wafer map pattern generators for UI demo purposes.

These patterns approximate the visual appearance of WM-811K defect classes
but are NOT derived from real fab data.  All outputs must be labelled
"SYNTHETIC" wherever displayed.

Wafer map encoding (matches WM-811K convention):
  0 = background  (outside the circular wafer die boundary)
  1 = good die    (passing)
  2 = defective die (failing)
"""

from __future__ import annotations

import numpy as np

from semiconductor_yield.config import WAFER_DEFECT_CLASSES


# ── Wafer geometry helpers ─────────────────────────────────────────────────────

def _wafer_mask(size: int = 64, radius_frac: float = 0.46) -> np.ndarray:
    """Boolean mask — True inside the circular wafer boundary."""
    cy, cx = (size - 1) / 2, (size - 1) / 2
    radius = size * radius_frac
    y, x = np.ogrid[:size, :size]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2


def _base_wafer(size: int = 64) -> np.ndarray:
    """All-good-die wafer: value=1 inside boundary, 0 outside."""
    wmap = np.zeros((size, size), dtype=np.float32)
    wmap[_wafer_mask(size)] = 1.0
    return wmap


# ── Per-class pattern generators ───────────────────────────────────────────────

def make_center_pattern(size: int = 64, seed: int | None = 42) -> np.ndarray:
    """Circular cluster of defects in the wafer center (~18% radius)."""
    rng = np.random.default_rng(seed)
    wmap = _base_wafer(size)
    cy, cx = (size - 1) / 2, (size - 1) / 2
    y, x = np.ogrid[:size, :size]
    center_zone = ((x - cx) ** 2 + (y - cy) ** 2) <= (size * 0.18) ** 2
    hit = rng.random((size, size)) < 0.88
    wmap[center_zone & _wafer_mask(size) & hit] = 2.0
    return wmap


def make_donut_pattern(size: int = 64, seed: int | None = 42) -> np.ndarray:
    """Annular defect band — defective ring between inner and outer radii."""
    rng = np.random.default_rng(seed)
    wmap = _base_wafer(size)
    cy, cx = (size - 1) / 2, (size - 1) / 2
    y, x = np.ogrid[:size, :size]
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    donut = (d2 >= (size * 0.15) ** 2) & (d2 <= (size * 0.33) ** 2)
    hit = rng.random((size, size)) < 0.82
    wmap[donut & _wafer_mask(size) & hit] = 2.0
    return wmap


def make_edge_ring_pattern(size: int = 64, seed: int | None = 42) -> np.ndarray:
    """Defect ring along the full wafer circumference (outer ~10% band)."""
    rng = np.random.default_rng(seed)
    wmap = _base_wafer(size)
    cy, cx = (size - 1) / 2, (size - 1) / 2
    y, x = np.ogrid[:size, :size]
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    wafer_r = size * 0.46
    edge = d2 >= (wafer_r - size * 0.10) ** 2
    hit = rng.random((size, size)) < 0.90
    wmap[edge & _wafer_mask(size) & hit] = 2.0
    return wmap


def make_edge_loc_pattern(size: int = 64, seed: int | None = 42) -> np.ndarray:
    """Localized defects on one sector of the wafer edge (bottom-left)."""
    rng = np.random.default_rng(seed)
    wmap = _base_wafer(size)
    cy, cx = (size - 1) / 2, (size - 1) / 2
    y, x = np.ogrid[:size, :size]
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    wafer_r = size * 0.46
    edge = d2 >= (wafer_r - size * 0.12) ** 2
    sector = (x < cx) & (y > cy)     # bottom-left quadrant
    hit = rng.random((size, size)) < 0.92
    wmap[edge & sector & _wafer_mask(size) & hit] = 2.0
    return wmap


def make_loc_pattern(size: int = 64, seed: int | None = 42) -> np.ndarray:
    """Small localized defect cluster off-centre."""
    rng = np.random.default_rng(seed)
    wmap = _base_wafer(size)
    cy_off, cx_off = int(size * 0.65), int(size * 0.38)
    y, x = np.ogrid[:size, :size]
    cluster = ((x - cx_off) ** 2 + (y - cy_off) ** 2) <= (size * 0.12) ** 2
    hit = rng.random((size, size)) < 0.87
    wmap[cluster & _wafer_mask(size) & hit] = 2.0
    return wmap


def make_near_full_pattern(size: int = 64, seed: int | None = 42) -> np.ndarray:
    """Nearly all dies defective (~90% fail rate inside boundary)."""
    rng = np.random.default_rng(seed)
    wmap = _base_wafer(size)
    hit = rng.random((size, size)) < 0.90
    wmap[_wafer_mask(size) & hit] = 2.0
    return wmap


def make_random_pattern(size: int = 64, seed: int | None = 42) -> np.ndarray:
    """Randomly scattered defects (~15% defect rate)."""
    rng = np.random.default_rng(seed)
    wmap = _base_wafer(size)
    hit = rng.random((size, size)) < 0.15
    wmap[_wafer_mask(size) & hit] = 2.0
    return wmap


def make_scratch_pattern(size: int = 64, seed: int | None = 42) -> np.ndarray:
    """Diagonal line defect (scratch), ±2 pixel width."""
    rng = np.random.default_rng(seed)
    wmap = _base_wafer(size)
    y_idx, x_idx = np.ogrid[:size, :size]
    offset = int(size * 0.04)
    line = np.abs(y_idx - x_idx - offset) <= 2
    hit = rng.random((size, size)) < 0.90
    wmap[line & _wafer_mask(size) & hit] = 2.0
    return wmap


def make_none_pattern(size: int = 64, seed: int | None = 42) -> np.ndarray:
    """Clean wafer — very few stray defects (~1% rate)."""
    rng = np.random.default_rng(seed)
    wmap = _base_wafer(size)
    stray = rng.random((size, size)) < 0.01
    wmap[_wafer_mask(size) & stray] = 2.0
    return wmap


# ── Registry ───────────────────────────────────────────────────────────────────

DEMO_GENERATORS: dict[str, object] = {
    "Center":    make_center_pattern,
    "Donut":     make_donut_pattern,
    "Edge-Ring": make_edge_ring_pattern,
    "Edge-Loc":  make_edge_loc_pattern,
    "Loc":       make_loc_pattern,
    "Near-full": make_near_full_pattern,
    "Random":    make_random_pattern,
    "Scratch":   make_scratch_pattern,
    "none":      make_none_pattern,
}

assert set(DEMO_GENERATORS) == set(WAFER_DEFECT_CLASSES), (
    "DEMO_GENERATORS keys must exactly match WAFER_DEFECT_CLASSES"
)


def generate_demo_sample(class_name: str, seed: int | None = 42) -> np.ndarray:
    """Return a synthetic wafer map for the given defect class.

    Args:
        class_name: One of WAFER_DEFECT_CLASSES.
        seed: Random seed for reproducibility.

    Returns:
        float32 array of shape (64, 64) with values in {0.0, 1.0, 2.0}.
        0 = background, 1 = good die, 2 = defective die.

    Raises:
        KeyError: class_name is not a known defect class.
    """
    if class_name not in DEMO_GENERATORS:
        raise KeyError(
            f"Unknown class '{class_name}'. Valid: {sorted(DEMO_GENERATORS)}"
        )
    return DEMO_GENERATORS[class_name](seed=seed)  # type: ignore[call-arg]
