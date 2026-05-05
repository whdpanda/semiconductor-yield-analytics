"""Wafer map preprocessing utilities (Module A).

Pure-function helpers for resizing, normalising, label encoding,
stratified splitting, and class-weight computation.

No model training here — only data transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit

from semiconductor_yield.config import RANDOM_SEED, WAFER_DEFECT_CLASSES, WAFER_MAP_SIZE
from semiconductor_yield.wafer.data_loader import IDX_TO_LABEL, LABEL_TO_IDX, WaferSample

# ── Resize ─────────────────────────────────────────────────────────────────────


def resize_wafer_map(
    wmap: np.ndarray,
    target_size: tuple[int, int] = WAFER_MAP_SIZE,
) -> np.ndarray:
    """Resize a 2-D wafer map to target_size using nearest-neighbour interpolation.

    Nearest-neighbour is mandatory here: the discrete values {0, 1, 2} encode
    background / good-die / defect-die. Bilinear or bicubic resampling would
    produce fractional values that corrupt this encoding.

    Args:
        wmap: 2-D array with values in {0, 1, 2} (any integer or float dtype).
        target_size: (height, width) in pixels.

    Returns:
        float32 array of shape target_size, values guaranteed in {0.0, 1.0, 2.0}.
    """
    from PIL import Image  # deferred — pillow is a listed dependency

    arr = np.asarray(wmap, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    img = img.resize((target_size[1], target_size[0]), resample=Image.Resampling.NEAREST)
    return np.array(img, dtype=np.float32)


# ── Normalisation ──────────────────────────────────────────────────────────────


def normalize_wafer_map(wmap: np.ndarray) -> np.ndarray:
    """Scale wafer map values from {0, 1, 2} to {0.0, 0.5, 1.0}.

    Deterministic per-class rescaling so values fall in [0, 1] without destroying
    the three-class structure. No statistics are estimated from the data.
    """
    return np.asarray(wmap, dtype=np.float32) / 2.0


# ── Label helpers ──────────────────────────────────────────────────────────────


def encode_label(label_name: str) -> int:
    """Return the integer index for a defect class name."""
    if label_name not in LABEL_TO_IDX:
        raise KeyError(
            f"Unknown class '{label_name}'. Valid: {sorted(LABEL_TO_IDX)}"
        )
    return LABEL_TO_IDX[label_name]


def decode_label(idx: int) -> str:
    """Return the class name for an integer index."""
    if idx not in IDX_TO_LABEL:
        raise KeyError(
            f"Unknown index {idx}. Valid indices: {sorted(IDX_TO_LABEL)}"
        )
    return IDX_TO_LABEL[idx]


# ── Stratified split ───────────────────────────────────────────────────────────


@dataclass
class SplitResult:
    train: list[WaferSample]
    val: list[WaferSample]
    test: list[WaferSample]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def stratified_split(
    samples: list[WaferSample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> SplitResult:
    """Split samples into train / val / test with class-stratification.

    Uses two-stage StratifiedShuffleSplit:
        Stage 1 — isolate test set from the full pool.
        Stage 2 — split the remaining pool into train and val.

    Limitation: This is a *sample-level* stratified split. In production, you
    would split at the lot level to prevent wafers from the same lot appearing in
    both train and test (data leakage). See docs/interview_notes.md for context.

    Args:
        samples: List of WaferSample objects. Each class must have ≥ 3 samples.
        train_ratio, val_ratio, test_ratio: Must sum to 1.0 (within 1e-6).
        seed: Random seed for reproducibility.

    Returns:
        SplitResult with .train, .val, .test list attributes.

    Raises:
        ValueError: Ratios don't sum to 1 or samples list is empty.
        ValueError (from sklearn): A class has too few samples to stratify.
    """
    if not samples:
        raise ValueError("samples list is empty")
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"train+val+test ratios must sum to 1.0, got {total_ratio:.6f}")

    indices = np.arange(len(samples))
    labels = np.array([s.label for s in samples])

    # Stage 1: isolate test set
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(indices, labels))

    # Stage 2: split trainval into train and val
    val_fraction = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_sub, val_sub = next(sss2.split(trainval_idx, labels[trainval_idx]))

    train_idx = trainval_idx[train_sub]
    val_idx = trainval_idx[val_sub]

    result = SplitResult(
        train=[samples[i] for i in train_idx],
        val=[samples[i] for i in val_idx],
        test=[samples[i] for i in test_idx],
    )
    logger.info(
        f"Stratified split → train={len(result.train):,}, "
        f"val={len(result.val):,}, test={len(result.test):,}"
    )
    return result


# ── Class imbalance utilities ──────────────────────────────────────────────────


def compute_class_weights(
    labels: Sequence[int],
    n_classes: int = len(WAFER_DEFECT_CLASSES),
) -> np.ndarray:
    """Compute balanced class weights for loss-function weighting.

    Formula: weight[c] = n_total / (n_classes × count[c])
    Identical to sklearn's compute_class_weight('balanced', ...).
    Classes with zero samples receive weight 0.0.

    Args:
        labels: Integer class indices.
        n_classes: Number of output classes.

    Returns:
        float64 array of shape (n_classes,), indexed by class index.
    """
    arr = np.asarray(labels)
    n = len(arr)
    weights = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        count = int((arr == c).sum())
        if count > 0:
            weights[c] = n / (n_classes * count)
    return weights


def imbalance_stats(samples: list[WaferSample]) -> pd.DataFrame:
    """Return class imbalance statistics as a tidy DataFrame.

    Columns:
        class_name      — defect pattern label
        count           — number of samples
        percentage      — count / total × 100
        imbalance_ratio — count_of_majority / count_of_this_class  (≥ 1.0)
        weight          — balanced class weight for loss weighting

    Rows are sorted descending by count.
    """
    if not samples:
        raise ValueError("samples list is empty")

    n = len(samples)
    counts: dict[str, int] = {c: 0 for c in WAFER_DEFECT_CLASSES}
    for s in samples:
        if s.label_name in counts:
            counts[s.label_name] += 1
        else:
            logger.warning(f"Unknown label_name '{s.label_name}' — excluded from stats")

    max_count = max(counts.values()) if counts else 1
    n_cls = len(WAFER_DEFECT_CLASSES)

    rows = [
        {
            "class_name":      name,
            "count":           cnt,
            "percentage":      round(cnt / n * 100, 2),
            "imbalance_ratio": round(max_count / cnt, 2) if cnt > 0 else float("inf"),
            "weight":          round(n / (n_cls * cnt), 4) if cnt > 0 else float("nan"),
        }
        for name, cnt in counts.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
