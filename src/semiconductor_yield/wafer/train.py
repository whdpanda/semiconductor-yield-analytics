"""Training pipeline for WaferCNN (Module A).

Public API:
  train_epoch(model, loader, optimizer, criterion, device) -> dict[str, float]
  validate_epoch(model, loader, criterion, device)         -> dict[str, float]
  fit(train_dataset, val_dataset, **kwargs)                -> dict
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from semiconductor_yield.config import MODELS_DIR, RANDOM_SEED, WAFER_DEFECT_CLASSES, WAFER_REPORTS_DIR
from semiconductor_yield.models.wafer_cnn import WaferCNN
from semiconductor_yield.wafer.dataset import (
    WaferMapDataset,
    make_balanced_subset,
    make_hybrid_subset,
    make_weighted_sampler,
)


# ── Per-epoch loops ────────────────────────────────────────────────────────────

def train_epoch(
    model: WaferCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch and return loss / accuracy / macro F1."""
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    n = len(all_labels)
    return {
        "loss":     total_loss / n,
        "accuracy": sum(p == l for p, l in zip(all_preds, all_labels)) / n,
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


def validate_epoch(
    model: WaferCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation epoch and return loss / accuracy / macro F1."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    n = len(all_labels)
    return {
        "loss":     total_loss / n,
        "accuracy": sum(p == l for p, l in zip(all_preds, all_labels)) / n,
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


# ── Full training loop ─────────────────────────────────────────────────────────

def fit(
    train_dataset: WaferMapDataset,
    val_dataset: WaferMapDataset,
    *,
    num_classes: int = 9,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    use_weighted_sampler: bool = True,
    class_weight_mode: str = "sqrt_inverse",
    balanced_subset: bool = False,
    samples_per_class: int = 500,
    hybrid_subset: bool = False,
    none_samples: int = 3000,
    major_class_samples: int = 1000,
    minor_class_samples: int = 500,
    rare_class_samples: int = 300,
    device_str: str = "auto",
    output_dir: Path | str = MODELS_DIR,
    report_dir: Path | str = WAFER_REPORTS_DIR,
    class_names: list[str] | None = None,
    run_id: str | None = None,
) -> dict:
    """Train WaferCNN and save the best checkpoint (by val macro F1).

    Args:
        train_dataset: Training split (augment=True recommended).
        val_dataset:   Validation split (augment=False).
        num_classes:   Number of output classes.
        epochs:        Training epochs.
        batch_size:    Mini-batch size for both train and val.
        lr:            Initial Adam learning rate.
        weight_decay:  L2 regularisation coefficient.
        dropout:       Dropout probability in classifier.
        use_weighted_sampler: When True and balanced_subset=False, apply
            WeightedRandomSampler using class_weight_mode.
        class_weight_mode: "sqrt_inverse" (default, gentler) or "inverse"
            (aggressive). "inverse" can cause single-class collapse on WM-811K.
        balanced_subset: If True, sub-sample at most samples_per_class examples
            per class and train with shuffle=True (no sampler needed).
        samples_per_class: Max samples per class when balanced_subset=True.
        hybrid_subset: If True, apply class-group-aware sampling (more none samples
            than individual defect classes) to reduce false alarms. Takes priority
            over balanced_subset.
        none_samples: Max none-class samples when hybrid_subset=True.
        major_class_samples: Max samples per major defect class.
        minor_class_samples: Max samples per minor defect class.
        rare_class_samples: Max samples per rare defect class.
        device_str:    ``"auto"`` picks CUDA if available, else CPU.
        output_dir:    Directory for model checkpoint.
        report_dir:    Root report directory; reports are written to
            report_dir/runs/<run_id>/.
        class_names:   Class name list, index-aligned. Defaults to WAFER_DEFECT_CLASSES.
        run_id:        Unique run identifier. Auto-generated from datetime if None.

    Returns:
        Metrics dict with keys: run_id, best_epoch, best_val_macro_f1, history, disclaimer.
    """
    if class_names is None:
        class_names = list(WAFER_DEFECT_CLASSES)

    torch.manual_seed(RANDOM_SEED)

    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"[WaferCNN] Training on {device}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    if hybrid_subset:
        hybrid_samples = make_hybrid_subset(
            train_dataset.samples,
            class_names=class_names,
            none_samples=none_samples,
            major_class_samples=major_class_samples,
            minor_class_samples=minor_class_samples,
            rare_class_samples=rare_class_samples,
            seed=RANDOM_SEED,
        )
        logger.info(
            f"[WaferCNN] Hybrid subset: {len(hybrid_samples):,} samples "
            f"(none={none_samples}, major={major_class_samples}, "
            f"minor={minor_class_samples}, rare={rare_class_samples})"
        )
        train_dataset = WaferMapDataset(hybrid_samples, augment=True)
        if use_weighted_sampler and class_weight_mode != "none":
            sampler = make_weighted_sampler(train_dataset, mode=class_weight_mode)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(RANDOM_SEED),
            )
    elif balanced_subset:
        balanced_samples = make_balanced_subset(
            train_dataset.samples, samples_per_class, seed=RANDOM_SEED
        )
        logger.info(
            f"[WaferCNN] Balanced subset: {len(balanced_samples):,} samples "
            f"({samples_per_class}/class max from {len(train_dataset):,} available)"
        )
        train_dataset = WaferMapDataset(balanced_samples, augment=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )
    elif use_weighted_sampler and class_weight_mode != "none":
        sampler = make_weighted_sampler(train_dataset, mode=class_weight_mode)
        logger.info(f"[WaferCNN] WeightedRandomSampler mode={class_weight_mode}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )

    n_train = len(train_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ── Model + loss + optimizer ───────────────────────────────────────────────
    model = WaferCNN(num_classes=num_classes, dropout=dropout).to(device)
    logger.info(f"[WaferCNN] {model.count_parameters():,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Run ID and output directories ─────────────────────────────────────────
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    run_report_dir = Path(report_dir) / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    run_report_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[WaferCNN] run_id={run_id}  reports → {run_report_dir}")

    history: dict[str, list] = {"train": [], "val": []}
    best_val_f1 = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / "wafer_cnn_best.pth"

    for epoch in range(1, epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train"].append(train_m)
        history["val"].append(val_m)

        improved = val_m["macro_f1"] > best_val_f1
        marker = " *" if improved else ""
        logger.info(
            f"[WaferCNN] epoch {epoch:3d}/{epochs}  "
            f"train loss={train_m['loss']:.4f} f1={train_m['macro_f1']:.4f}  "
            f"val loss={val_m['loss']:.4f} f1={val_m['macro_f1']:.4f}{marker}"
        )

        if improved:
            best_val_f1 = val_m["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    logger.info(
        f"[WaferCNN] Training complete. Best val macro F1={best_val_f1:.4f} at epoch {best_epoch}. "
        f"Checkpoint: {checkpoint_path}"
    )

    # ── Val reports on best model ──────────────────────────────────────────────
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        _save_val_reports(
            model, val_loader, device, class_names, run_report_dir,
            run_id=run_id, n_train=n_train,
        )

    # ── Save metrics ───────────────────────────────────────────────────────────
    if hybrid_subset:
        effective_weight_mode = class_weight_mode if (use_weighted_sampler and class_weight_mode != "none") else "none"
        sampling_mode = "hybrid"
    elif balanced_subset:
        effective_weight_mode = "none"
        sampling_mode = "balanced"
    elif use_weighted_sampler and class_weight_mode != "none":
        effective_weight_mode = class_weight_mode
        sampling_mode = "weighted"
    else:
        effective_weight_mode = "none"
        sampling_mode = "none"

    metrics = {
        "disclaimer": (
            "Metrics on WM-811K public dataset (portfolio project -- "
            "not real fab production performance)."
        ),
        "run_id": run_id,
        "training_config": {
            "epochs":               epochs,
            "batch_size":           batch_size,
            "lr":                   lr,
            "weight_decay":         weight_decay,
            "sampling_mode":        sampling_mode,
            "balanced_subset":      balanced_subset,
            "samples_per_class":    samples_per_class if balanced_subset else None,
            "hybrid_subset":        hybrid_subset,
            "none_samples":         none_samples if hybrid_subset else None,
            "major_class_samples":  major_class_samples if hybrid_subset else None,
            "minor_class_samples":  minor_class_samples if hybrid_subset else None,
            "rare_class_samples":   rare_class_samples if hybrid_subset else None,
            "class_weight_mode":    effective_weight_mode,
            "device":               str(device),
            "n_train":              n_train,
            "n_val":                len(val_dataset),
        },
        "best_epoch":        best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "history":           history,
    }
    metrics_path = run_report_dir / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[WaferCNN] Metrics saved to {metrics_path}")

    return metrics


# ── Internal helpers ───────────────────────────────────────────────────────────

def _save_val_reports(
    model: WaferCNN,
    val_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    run_report_dir: Path,
    run_id: str = "",
    n_train: int = 0,
) -> None:
    from semiconductor_yield.wafer.evaluate import (
        compute_fab_metrics,
        evaluate_model,
        plot_confusion_matrix,
        plot_confusion_matrix_normalized,
        save_classification_report,
        save_evaluation_summary,
    )

    result = evaluate_model(model, val_loader, device, class_names)

    cm_path = run_report_dir / "confusion_matrix_val.png"
    plot_confusion_matrix(result.confusion_matrix, class_names, cm_path, title="Confusion Matrix (Validation)")
    logger.info(f"[WaferCNN] Confusion matrix saved to {cm_path}")

    cm_norm_path = run_report_dir / "confusion_matrix_val_normalized.png"
    plot_confusion_matrix_normalized(
        result.confusion_matrix, class_names, cm_norm_path,
        title="Confusion Matrix Normalized (Validation)",
    )
    logger.info(f"[WaferCNN] Normalized confusion matrix saved to {cm_norm_path}")

    save_classification_report(result, run_report_dir)
    logger.info(f"[WaferCNN] Classification report saved to {run_report_dir}")

    save_evaluation_summary(
        result, run_report_dir, split_name="val",
        n_samples=len(result.y_true), run_id=run_id, n_train=n_train,
    )
    logger.info(f"[WaferCNN] Evaluation summary saved to {run_report_dir}")

    # ── evaluation_metrics.json (fab-aware metrics for comparison tool) ───────
    fab = compute_fab_metrics(result.y_true, result.y_pred, class_names)
    eval_metrics = {
        "disclaimer": (
            "Metrics on WM-811K public dataset (portfolio project -- "
            "not real fab production performance)."
        ),
        "run_id":         run_id,
        "split":          "val",
        "n_samples":      int(len(result.y_true)),
        "accuracy":       round(result.accuracy, 4),
        "macro_f1":       round(result.macro_f1, 4),
        "macro_precision": result.macro_precision,
        "macro_recall":    result.macro_recall,
        "weighted_f1":    result.weighted_f1,
        **fab,   # none_recall, false_alarm_rate, defect_recall, scratch_precision, scratch_recall
        "per_class": result.per_class,
    }
    eval_path = run_report_dir / "evaluation_metrics.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"[WaferCNN] Evaluation metrics saved to {eval_path}")

    # ── prediction_distribution.json ─────────────────────────────────────────
    n_classes = len(class_names)
    n_pred = max(len(result.y_pred), 1)
    pred_counts = np.bincount(result.y_pred, minlength=n_classes)
    true_counts = np.bincount(result.y_true, minlength=n_classes)
    dist = {
        "run_id": run_id,
        "split":  "val",
        "prediction_distribution": {
            class_names[i]: round(float(pred_counts[i] / n_pred), 4)
            for i in range(n_classes)
        },
        "true_distribution": {
            class_names[i]: round(float(true_counts[i] / n_pred), 4)
            for i in range(n_classes)
        },
    }
    dist_path = run_report_dir / "prediction_distribution.json"
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(dist, f, indent=2)
    logger.info(f"[WaferCNN] Prediction distribution saved to {dist_path}")
