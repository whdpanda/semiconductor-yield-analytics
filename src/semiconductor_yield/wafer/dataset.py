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

def make_weighted_sampler(dataset: WaferMapDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that equalises class frequency.

    Each sample receives weight 1 / class_count. The sampler draws
    len(dataset) samples with replacement per epoch, matching the dataset
    length so DataLoader iteration works the same as with shuffle=True.

    Prefer this over loss-weight class weighting when the imbalance is
    extreme (79% 'none' class in WM-811K), because it re-balances the
    gradient signal rather than just scaling the loss magnitude.
    """
    labels = dataset.get_labels()
    n_classes = max(labels) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)     # avoid division by zero
    class_weights = 1.0 / counts
    sample_weights = torch.tensor(
        [class_weights[lbl] for lbl in labels], dtype=torch.float32
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
