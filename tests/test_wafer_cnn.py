"""Tests for Module A: WaferMapDataset, WaferCNN, and training smoke tests.

All tests use synthetic WaferSample objects -- no WM-811K data required.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from semiconductor_yield.models.wafer_cnn import WaferCNN
from semiconductor_yield.wafer.data_loader import WaferSample
from semiconductor_yield.wafer.dataset import WaferMapDataset, make_weighted_sampler
from semiconductor_yield.wafer.train import fit, train_epoch, validate_epoch


# ── Shared fixtures ────────────────────────────────────────────────────────────

_LABEL_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Near-full", "Random", "Scratch", "none",
]


def _make_sample(label: int) -> WaferSample:
    """Synthetic 64×64 wafer map sample with discrete values in {0, 1, 2}."""
    wmap = np.random.randint(0, 3, size=(64, 64), dtype=np.uint8).astype(np.float32)
    return WaferSample(
        wafer_map=wmap,
        label=label,
        label_name=_LABEL_NAMES[label],
        lot_name="LOT001",
        wafer_index=0,
        split="train",
    )


def _make_dataset(n_per_class: int = 3, augment: bool = False) -> WaferMapDataset:
    """WaferMapDataset with n_per_class samples for each of 9 classes."""
    samples = [
        _make_sample(cls_idx)
        for cls_idx in range(9)
        for _ in range(n_per_class)
    ]
    return WaferMapDataset(samples, augment=augment)


# ── TestWaferDataset ──────────────────────────────────────────────────────────

class TestWaferDataset:
    def test_len(self):
        ds = _make_dataset(n_per_class=4)
        assert len(ds) == 36  # 9 classes × 4

    def test_getitem_returns_two_element_tuple(self):
        ds = _make_dataset()
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 2

    def test_tensor_shape(self):
        ds = _make_dataset()
        tensor, _ = ds[0]
        assert tensor.shape == (1, 64, 64), f"Expected (1, 64, 64), got {tensor.shape}"

    def test_tensor_dtype(self):
        ds = _make_dataset()
        tensor, _ = ds[0]
        assert tensor.dtype == torch.float32

    def test_label_dtype(self):
        ds = _make_dataset()
        _, label = ds[0]
        assert label.dtype == torch.long

    def test_values_in_unit_range(self):
        ds = _make_dataset()
        for i in range(len(ds)):
            tensor, _ = ds[i]
            assert float(tensor.min()) >= 0.0, "Values below 0.0"
            assert float(tensor.max()) <= 1.0, "Values above 1.0"

    def test_no_augment_deterministic(self):
        """Same index returns identical tensor when augment=False."""
        ds = _make_dataset(augment=False)
        t1, _ = ds[0]
        t2, _ = ds[0]
        assert torch.equal(t1, t2)

    def test_augment_callable(self):
        """augment=True runs without errors."""
        np.random.seed(0)
        ds = _make_dataset(n_per_class=2, augment=True)
        for i in range(len(ds)):
            tensor, label = ds[i]
            assert tensor.shape == (1, 64, 64)
            assert label.dtype == torch.long

    def test_labels_match_samples(self):
        ds = _make_dataset()
        for i, sample in enumerate(ds.samples):
            _, label = ds[i]
            assert label.item() == sample.label

    def test_get_labels_length(self):
        ds = _make_dataset(n_per_class=2)
        labels = ds.get_labels()
        assert len(labels) == 18

    def test_get_labels_are_ints(self):
        ds = _make_dataset()
        labels = ds.get_labels()
        assert all(isinstance(l, int) for l in labels)

    def test_weighted_sampler_num_samples(self):
        ds = _make_dataset(n_per_class=3)
        sampler = make_weighted_sampler(ds)
        assert sampler.num_samples == len(ds)

    def test_weighted_sampler_replacement(self):
        ds = _make_dataset(n_per_class=3)
        sampler = make_weighted_sampler(ds)
        assert sampler.replacement is True


# ── TestWaferCNN ──────────────────────────────────────────────────────────────

class TestWaferCNN:
    def test_forward_output_shape(self):
        model = WaferCNN(num_classes=9)
        x = torch.rand(4, 1, 64, 64)
        out = model(x)
        assert out.shape == (4, 9), f"Expected (4, 9), got {out.shape}"

    def test_forward_no_nan(self):
        model = WaferCNN(num_classes=9)
        x = torch.rand(8, 1, 64, 64)
        out = model(x)
        assert not torch.any(torch.isnan(out)), "NaN in model output"

    def test_batch_size_one(self):
        model = WaferCNN(num_classes=9)
        x = torch.rand(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 9)

    def test_parameter_count_in_range(self):
        """Architecture description claims ~94K params; allow ±25%."""
        model = WaferCNN(num_classes=9, dropout=0.3)
        n = model.count_parameters()
        assert 70_000 < n < 130_000, f"Unexpected param count: {n:,}"

    def test_logits_not_probabilities(self):
        """Output row sums should not equal 1.0 (no softmax in forward)."""
        model = WaferCNN(num_classes=9)
        model.eval()
        x = torch.rand(4, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        row_sums = out.sum(dim=1)
        # Softmax would force sum == 1.0; raw logits won't
        assert not torch.allclose(row_sums, torch.ones(4), atol=0.1), \
            "Output looks like probabilities — expected raw logits"

    def test_custom_num_classes(self):
        model = WaferCNN(num_classes=4, dropout=0.0)
        x = torch.rand(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 4)

    def test_eval_mode_deterministic(self):
        """In eval mode (dropout inactive), same input gives same output."""
        model = WaferCNN(num_classes=9, dropout=0.5)
        model.eval()
        x = torch.rand(2, 1, 64, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "Eval mode not deterministic"

    def test_train_mode_gradients(self):
        """Forward pass in train mode produces gradients on parameters."""
        model = WaferCNN(num_classes=9)
        model.train()
        x = torch.rand(2, 1, 64, 64)
        out = model(x)
        loss = out.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed after backward"


# ── TestTrainingSmoke ─────────────────────────────────────────────────────────

class TestTrainingSmoke:
    """Verify train/validate/fit run end-to-end on tiny synthetic data.

    No WM-811K data required — all samples are randomly generated.
    """

    def _loaders(self, n_per_class: int = 4, batch_size: int = 8):
        train_ds = _make_dataset(n_per_class=n_per_class, augment=True)
        val_ds   = _make_dataset(n_per_class=2, augment=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, train_ds, val_ds

    def test_train_epoch_returns_dict(self):
        model = WaferCNN(num_classes=9)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        loader, _, _, _ = self._loaders()
        m = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        assert {"loss", "accuracy", "macro_f1"} <= m.keys()

    def test_train_epoch_loss_positive(self):
        model = WaferCNN(num_classes=9)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        loader, _, _, _ = self._loaders()
        m = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        assert m["loss"] >= 0.0

    def test_train_epoch_metrics_bounded(self):
        model = WaferCNN(num_classes=9)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        loader, _, _, _ = self._loaders()
        m = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0

    def test_validate_epoch_returns_dict(self):
        model = WaferCNN(num_classes=9)
        criterion = torch.nn.CrossEntropyLoss()
        _, val_loader, _, _ = self._loaders()
        m = validate_epoch(model, val_loader, criterion, torch.device("cpu"))
        assert {"loss", "accuracy", "macro_f1"} <= m.keys()

    def test_validate_epoch_no_gradients(self):
        """validate_epoch must not accumulate gradients on parameters."""
        model = WaferCNN(num_classes=9)
        criterion = torch.nn.CrossEntropyLoss()
        _, val_loader, _, _ = self._loaders()
        validate_epoch(model, val_loader, criterion, torch.device("cpu"))
        for p in model.parameters():
            assert p.grad is None, f"Unexpected gradient after validate_epoch"

    def test_fit_saves_checkpoint(self, tmp_path):
        train_ds = _make_dataset(n_per_class=3)
        val_ds   = _make_dataset(n_per_class=2)
        fit(
            train_ds, val_ds,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            use_weighted_sampler=False,
            device_str="cpu",
            output_dir=tmp_path,
            report_dir=tmp_path,
        )
        assert (tmp_path / "wafer_cnn_best.pth").exists(), "Checkpoint not saved"

    def test_fit_saves_metrics_json(self, tmp_path):
        train_ds = _make_dataset(n_per_class=3)
        val_ds   = _make_dataset(n_per_class=2)
        fit(
            train_ds, val_ds,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            use_weighted_sampler=False,
            device_str="cpu",
            output_dir=tmp_path,
            report_dir=tmp_path,
        )
        metrics_path = tmp_path / "training_metrics.json"
        assert metrics_path.exists(), "training_metrics.json not saved"
        with open(metrics_path) as f:
            m = json.load(f)
        for key in ("best_epoch", "best_val_macro_f1", "disclaimer", "history"):
            assert key in m, f"Missing key in metrics: {key}"

    def test_fit_returns_metrics_dict(self, tmp_path):
        train_ds = _make_dataset(n_per_class=3)
        val_ds   = _make_dataset(n_per_class=2)
        result = fit(
            train_ds, val_ds,
            epochs=2,
            batch_size=8,
            lr=1e-3,
            use_weighted_sampler=False,
            device_str="cpu",
            output_dir=tmp_path,
            report_dir=tmp_path,
        )
        assert result["best_epoch"] in (1, 2)
        assert 0.0 <= result["best_val_macro_f1"] <= 1.0
        assert len(result["history"]["train"]) == 2
        assert len(result["history"]["val"]) == 2

    def test_fit_with_weighted_sampler(self, tmp_path):
        """WeightedRandomSampler path should also complete without error."""
        train_ds = _make_dataset(n_per_class=4)
        val_ds   = _make_dataset(n_per_class=2)
        result = fit(
            train_ds, val_ds,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            use_weighted_sampler=True,
            device_str="cpu",
            output_dir=tmp_path,
            report_dir=tmp_path,
        )
        assert (tmp_path / "wafer_cnn_best.pth").exists()
