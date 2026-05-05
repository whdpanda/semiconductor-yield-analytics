"""Lightweight CNN baseline for wafer map defect classification (Module A).

Architecture: 3 convolutional blocks + Global Average Pooling + linear head.
~94 K trainable parameters. Runs comfortably on CPU.

Design rationale:
  - Three conv blocks capture local → mid-level → coarse spatial features.
  - Global Average Pooling (GAP) instead of large fully-connected layers:
    reduces parameters from ~8 M (FC) to ~1 K, retains spatial awareness.
  - BatchNorm after each conv for stable training without careful LR tuning.
  - Single dropout before the classifier to prevent overfitting on small splits.

Input:  (batch, 1, 64, 64)  — single-channel, normalised to [0, 1]
Output: (batch, num_classes) — raw logits; apply softmax for probabilities.

This is a portfolio baseline, not a production model. See README for context.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WaferCNN(nn.Module):
    """3-block convolutional wafer map classifier with Global Average Pooling.

    Args:
        num_classes: Number of output classes (9 for WM-811K).
        dropout: Dropout probability before the linear head (default 0.3).
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (1, 64, 64) → (32, 32, 32)
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (32, 32, 32) → (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (64, 16, 16) → (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling: (128, 8, 8) → (128, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),              # (128, 1, 1) → (128,)
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape (B, 1, H, W), values in [0, 1].

        Returns:
            Logit tensor of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
