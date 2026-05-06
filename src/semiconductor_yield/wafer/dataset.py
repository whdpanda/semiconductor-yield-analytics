"""PyTorch Dataset for WM-811K wafer map classification (Module A).

Wraps a list of WaferSample objects. Each item returns:
  (tensor, label) where tensor is float32 shape (1, H, W), values in [0, 1].

Augmentation exploits wafer rotational symmetry (ring defects look identical
after 90° rotation and horizontal/vertical flip).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from semiconductor_yield.config import RANDOM_SEED
from semiconductor_yield.wafer.data_loader import WaferSample
from semiconductor_yield.wafer.preprocess import normalize_wafer_map


class WaferMapDataset(Dataset):
    """PyTorch Dataset wrapping a list of WaferSample objects.

    Args:
        samples: Pre-split list of WaferSample (from stratified_split).
        augment: When True, applies random rot90 / flip augmentation. Use
            True for training splits, False for val/test splits.
    """

    def __init__(self, samples: list[WaferSample], augment: bool = False) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        wmap = normalize_wafer_map(sample.wafer_map)  # float32, [0.0, 1.0]

        if self.augment:
            wmap = _random_augment(wmap)

        # (H, W) → (1, H, W)
        tensor = torch.from_numpy(np.ascontiguousarray(wmap)).unsqueeze(0)
        label = torch.tensor(sample.label, dtype=torch.long)
        return tensor, label

    def get_labels(self) -> list[int]:
        """Return all integer labels; used by make_weighted_sampler."""
        return [s.label for s in self.samples]


# ── Augmentation ───────────────────────────────────────────────────────────────

def _random_augment(wmap: np.ndarray) -> np.ndarray:
    """Training-time augmentation exploiting wafer rotational symmetry.

    All transforms preserve the discrete {0, 1, 2} value structure.
    """
    # Random 90° rotation (0 / 90 / 180 / 270°)
    k = random.randint(0, 3)
    if k:
        wmap = np.rot90(wmap, k=k)

    # Random horizontal flip
    if random.random() < 0.5:
        wmap = np.fliplr(wmap)

    # Random vertical flip
    if random.random() < 0.5:
        wmap = np.flipud(wmap)

    return np.ascontiguousarray(wmap)


# ── Imbalance handling ─────────────────────────────────────────────────────────

def make_weighted_sampler(
    dataset: WaferMapDataset,
    mode: str = "sqrt_inverse",
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler with a configurable class-weight strategy.

    Args:
        dataset: WaferMapDataset with a .get_labels() method.
        mode: How per-sample weights are computed from class counts:
            - "inverse"      : weight = 1 / count   (aggressive; ~123× for WM-811K)
            - "sqrt_inverse" : weight = 1 / sqrt(count)  (gentler default; ~11× for WM-811K)

    Returns:
        WeightedRandomSampler drawing len(dataset) samples with replacement.

    Note on WM-811K:
        "inverse" causes severe over-sampling of rare classes like Scratch (~775
        train samples appear ~17× per epoch), which can lead to single-class
        collapse. "sqrt_inverse" reduces the replication to ~7× while still
        improving minority-class recall meaningfully.
    """
    if mode not in ("inverse", "sqrt_inverse"):
        raise ValueError(f"Unknown mode '{mode}'. Valid: 'inverse', 'sqrt_inverse'.")

    labels = dataset.get_labels()
    n_classes = max(labels) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)

    if mode == "sqrt_inverse":
        class_weights = 1.0 / np.sqrt(counts)
    else:
        class_weights = 1.0 / counts

    sample_weights = torch.tensor(
        [class_weights[lbl] for lbl in labels], dtype=torch.float32
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_balanced_subset(
    samples: list[WaferSample],
    samples_per_class: int = 500,
    seed: int = RANDOM_SEED,
) -> list[WaferSample]:
    """Select at most samples_per_class examples from each class.

    Classes with fewer than samples_per_class samples contribute all their
    samples. The resulting dataset is ~balanced and can be used with
    shuffle=True instead of WeightedRandomSampler.

    Args:
        samples: Full list of training samples (pre-split).
        samples_per_class: Maximum samples drawn from each class.
        seed: RNG seed for reproducible selection.

    Returns:
        Balanced subset with at most samples_per_class samples per class.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[WaferSample]] = {}
    for s in samples:
        by_class.setdefault(s.label, []).append(s)

    result: list[WaferSample] = []
    for cls_idx in sorted(by_class):
        cls_samples = by_class[cls_idx]
        if len(cls_samples) <= samples_per_class:
            result.extend(cls_samples)
        else:
            chosen = rng.choice(len(cls_samples), size=samples_per_class, replace=False)
            result.extend(cls_samples[i] for i in chosen)

    return result
