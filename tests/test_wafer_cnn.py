"""Tests for Module A: WaferMapDataset, WaferCNN, and training smoke tests.

All tests use synthetic WaferSample objects -- no WM-811K data required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from semiconductor_yield.models.wafer_cnn import WaferCNN
from semiconductor_yield.wafer.data_loader import WaferSample
from semiconductor_yield.wafer.dataset import (
    WaferMapDataset,
    make_balanced_subset,
    make_hybrid_subset,
    make_weighted_sampler,
)
from semiconductor_yield.wafer.evaluate import (
    apply_none_bias_threshold,
    collect_probabilities,
    compute_fab_metrics,
    save_calibration_report,
)
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

    def test_weighted_sampler_sqrt_inverse_mode(self):
        ds = _make_dataset(n_per_class=3)
        sampler = make_weighted_sampler(ds, mode="sqrt_inverse")
        assert sampler.num_samples == len(ds)

    def test_weighted_sampler_inverse_mode(self):
        ds = _make_dataset(n_per_class=3)
        sampler = make_weighted_sampler(ds, mode="inverse")
        assert sampler.num_samples == len(ds)

    def test_weighted_sampler_sqrt_weights_less_extreme(self):
        """sqrt_inverse weights should be less extreme than inverse weights."""
        import torch
        ds = _make_dataset(n_per_class=3)
        s_inv  = make_weighted_sampler(ds, mode="inverse")
        s_sqrt = make_weighted_sampler(ds, mode="sqrt_inverse")
        ratio_inv  = float(s_inv.weights.max()  / s_inv.weights.min())
        ratio_sqrt = float(s_sqrt.weights.max() / s_sqrt.weights.min())
        assert ratio_sqrt <= ratio_inv, "sqrt_inverse should be less extreme than inverse"

    def test_weighted_sampler_invalid_mode(self):
        ds = _make_dataset(n_per_class=3)
        with pytest.raises(ValueError, match="Unknown mode"):
            make_weighted_sampler(ds, mode="bad_mode")

    def test_make_balanced_subset_cap(self):
        """Each class should have at most samples_per_class items."""
        samples = _make_dataset(n_per_class=10).samples
        subset = make_balanced_subset(samples, samples_per_class=5, seed=0)
        from collections import Counter
        counts = Counter(s.label for s in subset)
        assert all(v <= 5 for v in counts.values())

    def test_make_balanced_subset_small_class(self):
        """Classes with fewer samples than the cap keep all their samples."""
        samples = _make_dataset(n_per_class=2).samples
        subset = make_balanced_subset(samples, samples_per_class=100, seed=0)
        assert len(subset) == len(samples)

    def test_make_balanced_subset_reproducible(self):
        samples = _make_dataset(n_per_class=10).samples
        s1 = [s.label for s in make_balanced_subset(samples, 5, seed=7)]
        s2 = [s.label for s in make_balanced_subset(samples, 5, seed=7)]
        assert s1 == s2

    def test_fit_balanced_subset(self, tmp_path):
        """balanced_subset=True path should complete without error."""
        train_ds = _make_dataset(n_per_class=10)
        val_ds   = _make_dataset(n_per_class=2)
        result = fit(
            train_ds, val_ds,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            balanced_subset=True,
            samples_per_class=5,
            device_str="cpu",
            output_dir=tmp_path,
            report_dir=tmp_path,
            run_id="test_balanced",
        )
        assert (tmp_path / "wafer_cnn_best.pth").exists()
        assert result["run_id"] == "test_balanced"


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
            run_id="test_run",
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
            run_id="test_run",
        )
        metrics_path = tmp_path / "runs" / "test_run" / "training_metrics.json"
        assert metrics_path.exists(), "training_metrics.json not saved"
        with open(metrics_path) as f:
            m = json.load(f)
        for key in ("run_id", "best_epoch", "best_val_macro_f1", "disclaimer", "history"):
            assert key in m, f"Missing key in metrics: {key}"
        assert m["run_id"] == "test_run"

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
            run_id="test_run",
        )
        assert result["run_id"] == "test_run"
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
            run_id="test_run",
        )
        assert (tmp_path / "wafer_cnn_best.pth").exists()


# ── TestThresholdCalibration ──────────────────────────────────────────────────

class TestThresholdCalibration:
    """Tests for apply_none_bias_threshold and save_calibration_report."""

    _NONE_IDX = 8
    _SCRATCH_IDX = 7
    _N_CLASSES = 9
    _CLASS_NAMES = [
        "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
        "Near-full", "Random", "Scratch", "none",
    ]

    def _uniform_probs(self, n: int) -> np.ndarray:
        """Uniform probability array — argmax is always class 0."""
        p = np.ones((n, self._N_CLASSES)) / self._N_CLASSES
        return p

    def _high_defect_probs(self, n: int, defect_idx: int, confidence: float) -> np.ndarray:
        """Probs concentrated on one defect class."""
        p = np.zeros((n, self._N_CLASSES))
        p[:, defect_idx] = confidence
        p[:, self._NONE_IDX] = 1.0 - confidence
        return p

    def _high_none_probs(self, n: int, confidence: float = 0.90) -> np.ndarray:
        """Probs concentrated on none class."""
        p = np.zeros((n, self._N_CLASSES))
        p[:, self._NONE_IDX] = confidence
        p[:, 0] = 1.0 - confidence
        return p

    def test_threshold_zero_matches_argmax(self):
        """threshold=0.0 must produce identical predictions to argmax."""
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(self._N_CLASSES), size=50)
        preds_argmax = probs.argmax(axis=1)
        preds_cal = apply_none_bias_threshold(probs, self._NONE_IDX, 0.0)
        np.testing.assert_array_equal(preds_argmax, preds_cal)

    def test_threshold_one_predicts_all_none(self):
        """threshold=1.0 overrides every prediction to none (probs < 1)."""
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(self._N_CLASSES), size=30)
        preds = apply_none_bias_threshold(probs, self._NONE_IDX, 1.0)
        assert (preds == self._NONE_IDX).all()

    def test_high_confidence_defect_survives_threshold(self):
        """A defect with 0.95 confidence should survive threshold=0.5."""
        probs = self._high_defect_probs(10, defect_idx=0, confidence=0.95)
        preds = apply_none_bias_threshold(probs, self._NONE_IDX, 0.5)
        assert (preds == 0).all()

    def test_low_confidence_defect_suppressed(self):
        """A defect with 0.20 confidence should be suppressed at threshold=0.5."""
        probs = self._high_defect_probs(10, defect_idx=0, confidence=0.20)
        preds = apply_none_bias_threshold(probs, self._NONE_IDX, 0.5)
        assert (preds == self._NONE_IDX).all()

    def test_high_none_prob_unchanged_by_threshold(self):
        """If none has highest probability, threshold never changes the prediction."""
        probs = self._high_none_probs(10, confidence=0.90)
        for t in [0.0, 0.5, 0.8]:
            preds = apply_none_bias_threshold(probs, self._NONE_IDX, t)
            assert (preds == self._NONE_IDX).all(), f"Failed at threshold={t}"

    def test_higher_threshold_increases_none_fraction(self):
        """As threshold increases, more predictions become none."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(self._N_CLASSES), size=200)
        none_fracs = []
        for t in [0.0, 0.3, 0.5, 0.7, 0.9]:
            preds = apply_none_bias_threshold(probs, self._NONE_IDX, t)
            none_fracs.append((preds == self._NONE_IDX).mean())
        # None fraction should be non-decreasing as threshold increases
        for i in range(len(none_fracs) - 1):
            assert none_fracs[i] <= none_fracs[i + 1], (
                f"none fraction decreased from t={[0.0, 0.3, 0.5, 0.7, 0.9][i]} "
                f"to t={[0.0, 0.3, 0.5, 0.7, 0.9][i+1]}"
            )

    def test_save_calibration_report_creates_all_files(self, tmp_path):
        """save_calibration_report creates all 4 required output files."""
        rng = np.random.default_rng(0)
        n = 90  # 10 per class
        probs_cal  = rng.dirichlet(np.ones(self._N_CLASSES), size=n)
        y_cal      = np.repeat(np.arange(self._N_CLASSES), 10)
        probs_eval = rng.dirichlet(np.ones(self._N_CLASSES), size=n)
        y_eval     = np.repeat(np.arange(self._N_CLASSES), 10)

        save_calibration_report(
            probs_cal=probs_cal,
            y_true_cal=y_cal,
            probs_eval=probs_eval,
            y_true_eval=y_eval,
            class_names=self._CLASS_NAMES,
            output_dir=tmp_path,
            thresholds=[0.0, 0.5, 0.9],
        )

        assert (tmp_path / "calibration_report.csv").exists()
        assert (tmp_path / "calibration_summary.json").exists()
        assert (tmp_path / "confusion_matrix_calibrated.png").exists()
        assert (tmp_path / "classification_report_calibrated.csv").exists()
        assert (tmp_path / "classification_report_calibrated.json").exists()

    def test_save_calibration_report_returns_recommended_threshold(self, tmp_path):
        rng = np.random.default_rng(1)
        n = 90
        probs = rng.dirichlet(np.ones(self._N_CLASSES), size=n)
        y = np.repeat(np.arange(self._N_CLASSES), 10)
        result = save_calibration_report(
            probs_cal=probs, y_true_cal=y,
            probs_eval=probs, y_true_eval=y,
            class_names=self._CLASS_NAMES,
            output_dir=tmp_path,
            thresholds=[0.0, 0.5, 0.9],
        )
        assert "recommended_threshold" in result
        assert result["recommended_threshold"] in [0.0, 0.5, 0.9]

    def test_calibration_report_csv_has_required_columns(self, tmp_path):
        rng = np.random.default_rng(2)
        n = 90
        probs = rng.dirichlet(np.ones(self._N_CLASSES), size=n)
        y = np.repeat(np.arange(self._N_CLASSES), 10)
        save_calibration_report(
            probs_cal=probs, y_true_cal=y,
            probs_eval=probs, y_true_eval=y,
            class_names=self._CLASS_NAMES,
            output_dir=tmp_path,
            thresholds=[0.0, 0.5],
        )
        import csv as csv_mod
        with open(tmp_path / "calibration_report.csv") as f:
            reader = csv_mod.DictReader(f)
            cols = reader.fieldnames
        required = {
            "threshold", "accuracy", "macro_f1", "weighted_f1",
            "none_recall", "scratch_precision", "scratch_recall",
            "false_alarm_rate", "defect_recall",
        }
        assert required.issubset(set(cols)), f"Missing columns: {required - set(cols)}"

    def test_compute_fab_metrics_returns_required_keys(self):
        """compute_fab_metrics must include false_alarm_rate, defect_recall, none_recall."""
        rng = np.random.default_rng(5)
        n = 90
        y_true = np.repeat(np.arange(9), 10)
        y_pred = rng.integers(0, 9, size=n)
        result = compute_fab_metrics(y_true, y_pred, self._CLASS_NAMES)
        required = {
            "accuracy", "macro_f1", "weighted_f1",
            "none_recall", "false_alarm_rate", "defect_recall",
            "scratch_precision", "scratch_recall",
        }
        assert required.issubset(set(result.keys())), (
            f"Missing keys: {required - set(result.keys())}"
        )

    def test_compute_fab_metrics_false_alarm_rate_range(self):
        """false_alarm_rate must be in [0, 1]."""
        rng = np.random.default_rng(6)
        y_true = np.repeat(np.arange(9), 10)
        y_pred = rng.integers(0, 9, size=90)
        result = compute_fab_metrics(y_true, y_pred, self._CLASS_NAMES)
        assert 0.0 <= result["false_alarm_rate"] <= 1.0

    def test_compute_fab_metrics_all_none_predictions(self):
        """Predicting all none: false_alarm_rate=0, defect_recall=0."""
        y_true = np.array([0, 1, 7, 8, 8])  # mix of classes
        y_pred = np.full(5, 8)               # all predicted as none (idx=8)
        result = compute_fab_metrics(y_true, y_pred, self._CLASS_NAMES)
        assert result["false_alarm_rate"] == pytest.approx(0.0)
        assert result["defect_recall"] == pytest.approx(0.0)

    def test_calibration_summary_json_has_required_keys(self, tmp_path):
        rng = np.random.default_rng(3)
        n = 90
        probs = rng.dirichlet(np.ones(self._N_CLASSES), size=n)
        y = np.repeat(np.arange(self._N_CLASSES), 10)
        save_calibration_report(
            probs_cal=probs, y_true_cal=y,
            probs_eval=probs, y_true_eval=y,
            class_names=self._CLASS_NAMES,
            output_dir=tmp_path,
            thresholds=[0.0, 0.5],
        )
        import json as json_mod
        with open(tmp_path / "calibration_summary.json") as f:
            s = json_mod.load(f)
        required_keys = {
            "recommended_threshold",
            "baseline_none_recall", "calibrated_none_recall",
            "baseline_false_alarm_rate", "calibrated_false_alarm_rate",
            "baseline_scratch_precision", "calibrated_scratch_precision",
            "tradeoff_note",
        }
        assert required_keys.issubset(set(s.keys())), (
            f"Missing keys: {required_keys - set(s.keys())}"
        )


# ── TestHybridSubset ──────────────────────────────────────────────────────────

class TestHybridSubset:
    """Tests for make_hybrid_subset."""

    _CLASS_NAMES = [
        "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
        "Near-full", "Random", "Scratch", "none",
    ]

    def _make_samples(self, counts: dict[str, int]) -> list[WaferSample]:
        """Build a synthetic sample list with specified per-class counts."""
        result = []
        for name, n in counts.items():
            idx = self._CLASS_NAMES.index(name)
            for i in range(n):
                wmap = np.zeros((16, 16), dtype=np.float32)
                result.append(WaferSample(
                    wafer_map=wmap, label=idx, label_name=name,
                    lot_name="LOT001", wafer_index=i, split="train",
                ))
        return result

    def test_none_class_exceeds_any_single_defect_class(self):
        """none count must be >= any individual defect class count."""
        samples = self._make_samples({
            "none": 5000, "Edge-Ring": 2000, "Edge-Loc": 1500,
            "Center": 1200, "Loc": 800, "Scratch": 400,
            "Random": 600, "Donut": 150, "Near-full": 100,
        })
        result = make_hybrid_subset(
            samples, self._CLASS_NAMES,
            none_samples=3000, major_class_samples=1000,
            minor_class_samples=500, rare_class_samples=300,
        )
        by_name: dict[str, int] = {}
        for s in result:
            by_name[self._CLASS_NAMES[s.label]] = by_name.get(self._CLASS_NAMES[s.label], 0) + 1
        none_count = by_name.get("none", 0)
        for name, cnt in by_name.items():
            if name != "none":
                assert none_count >= cnt, (
                    f"none ({none_count}) < {name} ({cnt})"
                )

    def test_caps_are_respected(self):
        """Each class must not exceed its group cap."""
        samples = self._make_samples({
            "none": 5000, "Edge-Ring": 2000, "Scratch": 800,
            "Near-full": 400,
        })
        result = make_hybrid_subset(
            samples, self._CLASS_NAMES,
            none_samples=2000, major_class_samples=500,
            minor_class_samples=300, rare_class_samples=200,
        )
        by_name: dict[str, int] = {}
        for s in result:
            by_name[self._CLASS_NAMES[s.label]] = by_name.get(self._CLASS_NAMES[s.label], 0) + 1

        assert by_name.get("none",     0) <= 2000
        assert by_name.get("Edge-Ring",0) <= 500
        assert by_name.get("Scratch",  0) <= 300
        assert by_name.get("Near-full",0) <= 200

    def test_class_below_cap_keeps_all(self):
        """Classes with fewer samples than the cap should keep all their samples."""
        samples = self._make_samples({"none": 100, "Edge-Ring": 50})
        result = make_hybrid_subset(
            samples, self._CLASS_NAMES,
            none_samples=5000, major_class_samples=5000,
        )
        by_name: dict[str, int] = {}
        for s in result:
            by_name[self._CLASS_NAMES[s.label]] = by_name.get(self._CLASS_NAMES[s.label], 0) + 1
        assert by_name.get("none",     0) == 100
        assert by_name.get("Edge-Ring",0) == 50

    def test_is_reproducible(self):
        """Same seed must produce identical subsets."""
        samples = self._make_samples({
            "none": 2000, "Edge-Ring": 1500, "Scratch": 600,
        })
        r1 = make_hybrid_subset(samples, self._CLASS_NAMES, seed=42)
        r2 = make_hybrid_subset(samples, self._CLASS_NAMES, seed=42)
        assert [s.label for s in r1] == [s.label for s in r2]


# ── TestCompareWaferRuns ──────────────────────────────────────────────────────

class TestCompareWaferRuns:
    """Tests for compare_wafer_runs logic."""

    _METRICS_KEYS = [
        "accuracy", "macro_f1", "weighted_f1",
        "none_recall", "false_alarm_rate", "defect_recall",
        "scratch_precision", "scratch_recall",
    ]

    def _write_metrics(self, tmp_path: Path, run_id: str, values: dict) -> None:
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id, "split": "val",
            "disclaimer": "test",
            **values,
        }
        with open(run_dir / "evaluation_metrics.json", "w") as f:
            import json as _j
            _j.dump(payload, f)

    def test_compare_returns_overall_improved(self, tmp_path):
        """Candidate that improves none_recall and lowers false_alarm_rate → improved."""
        self._write_metrics(tmp_path, "base", {
            "accuracy": 0.50, "macro_f1": 0.40, "weighted_f1": 0.45,
            "none_recall": 0.40, "false_alarm_rate": 0.60,
            "defect_recall": 0.85, "scratch_precision": 0.01, "scratch_recall": 0.50,
        })
        self._write_metrics(tmp_path, "cand", {
            "accuracy": 0.55, "macro_f1": 0.45, "weighted_f1": 0.50,
            "none_recall": 0.80, "false_alarm_rate": 0.20,
            "defect_recall": 0.70, "scratch_precision": 0.15, "scratch_recall": 0.40,
        })
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from compare_wafer_runs import compare
        result = compare("base", "cand", reports_root=tmp_path)
        assert result["overall_verdict"] == "improved"

    def test_compare_saves_json_and_csv(self, tmp_path):
        """save_comparison must create .json and .csv files."""
        self._write_metrics(tmp_path, "base2", {
            "accuracy": 0.50, "macro_f1": 0.40, "weighted_f1": 0.45,
            "none_recall": 0.40, "false_alarm_rate": 0.60,
            "defect_recall": 0.85, "scratch_precision": 0.01, "scratch_recall": 0.50,
        })
        self._write_metrics(tmp_path, "cand2", {
            "accuracy": 0.45, "macro_f1": 0.35, "weighted_f1": 0.40,
            "none_recall": 0.30, "false_alarm_rate": 0.70,
            "defect_recall": 0.80, "scratch_precision": 0.01, "scratch_recall": 0.45,
        })
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from compare_wafer_runs import compare, save_comparison
        result = compare("base2", "cand2", reports_root=tmp_path)
        json_p, csv_p = save_comparison(result, tmp_path)
        assert json_p.exists()
        assert csv_p.exists()

    def test_compare_metrics_has_required_keys(self, tmp_path):
        self._write_metrics(tmp_path, "b3", {
            k: 0.5 for k in self._METRICS_KEYS
        })
        self._write_metrics(tmp_path, "c3", {
            k: 0.5 for k in self._METRICS_KEYS
        })
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from compare_wafer_runs import compare
        result = compare("b3", "c3", reports_root=tmp_path)
        for m in self._METRICS_KEYS:
            assert m in result["metrics"], f"Missing metric: {m}"
            assert "verdict" in result["metrics"][m]

    def test_fit_saves_evaluation_metrics_json(self, tmp_path):
        """fit() must save evaluation_metrics.json with none_recall, false_alarm_rate."""
        train_ds = _make_dataset(n_per_class=3)
        val_ds   = _make_dataset(n_per_class=2)
        fit(
            train_ds, val_ds,
            epochs=1, batch_size=8, lr=1e-3,
            use_weighted_sampler=False,
            device_str="cpu",
            output_dir=tmp_path, report_dir=tmp_path,
            run_id="test_eval_metrics",
        )
        eval_path = tmp_path / "runs" / "test_eval_metrics" / "evaluation_metrics.json"
        assert eval_path.exists(), "evaluation_metrics.json not created by fit()"
        import json as _j
        with open(eval_path) as f:
            m = _j.load(f)
        for key in ("none_recall", "false_alarm_rate", "defect_recall",
                    "scratch_precision", "scratch_recall"):
            assert key in m, f"Missing key in evaluation_metrics.json: {key}"

    def test_fit_hybrid_mode_produces_subset(self, tmp_path):
        """fit() with hybrid_subset=True should complete and save outputs."""
        train_ds = _make_dataset(n_per_class=10)
        val_ds   = _make_dataset(n_per_class=2)
        result = fit(
            train_ds, val_ds,
            epochs=1, batch_size=8, lr=1e-3,
            use_weighted_sampler=False,
            hybrid_subset=True,
            none_samples=5,
            major_class_samples=3,
            minor_class_samples=2,
            rare_class_samples=2,
            device_str="cpu",
            output_dir=tmp_path, report_dir=tmp_path,
            run_id="test_hybrid",
        )
        assert result["training_config"]["sampling_mode"] == "hybrid"
        assert (tmp_path / "wafer_cnn_best.pth").exists()
