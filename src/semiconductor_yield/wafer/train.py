"""Training pipeline for WaferCNN (Module A).

Public API:
  train_epoch(model, loader, optimizer, criterion, device) -> dict[str, float]
  validate_epoch(model, loader, criterion, device)         -> dict[str, float]
  fit(train_dataset, val_dataset, **kwargs)                -> dict
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from semiconductor_yield.config import MODELS_DIR, RANDOM_SEED, WAFER_DEFECT_CLASSES, WAFER_REPORTS_DIR
from semiconductor_yield.models.wafer_cnn import WaferCNN
from semiconductor_yield.wafer.dataset import WaferMapDataset, make_weighted_sampler


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
    device_str: str = "auto",
    output_dir: Path | str = MODELS_DIR,
    report_dir: Path | str = WAFER_REPORTS_DIR,
    class_names: list[str] | None = None,
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
        use_weighted_sampler: If True, use WeightedRandomSampler to equalise
            class frequency in training. Preferred over loss class-weights for
            extreme imbalance like WM-811K's 79% 'none' class.
        device_str:    ``"auto"`` picks CUDA if available, else CPU.
        output_dir:    Directory for model checkpoint.
        report_dir:    Directory for training_metrics.json and confusion matrix.
        class_names:   Class name list, index-aligned. Defaults to WAFER_DEFECT_CLASSES.

    Returns:
        Metrics dict with keys: best_epoch, best_val_macro_f1, history, disclaimer.
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
    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ── Model + loss + optimizer ───────────────────────────────────────────────
    model = WaferCNN(num_classes=num_classes, dropout=dropout).to(device)
    logger.info(f"[WaferCNN] {model.count_parameters():,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ──────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Confusion matrix on val set (best model) ───────────────────────────────
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        _save_val_confusion_matrix(model, val_loader, device, class_names, report_dir)

    # ── Save metrics ───────────────────────────────────────────────────────────
    metrics = {
        "disclaimer": (
            "Metrics on WM-811K public dataset (portfolio project — "
            "not real fab production performance)."
        ),
        "training_config": {
            "epochs":               epochs,
            "batch_size":           batch_size,
            "lr":                   lr,
            "weight_decay":         weight_decay,
            "use_weighted_sampler": use_weighted_sampler,
            "device":               str(device),
            "n_train":              len(train_dataset),
            "n_val":                len(val_dataset),
        },
        "best_epoch":        best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "history":           history,
    }
    metrics_path = report_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[WaferCNN] Metrics saved to {metrics_path}")

    return metrics


# ── Internal helpers ───────────────────────────────────────────────────────────

def _save_val_confusion_matrix(
    model: WaferCNN,
    val_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    report_dir: Path,
) -> None:
    from semiconductor_yield.wafer.evaluate import evaluate_model, plot_confusion_matrix

    result = evaluate_model(model, val_loader, device, class_names)
    cm_path = report_dir / "confusion_matrix_val.png"
    plot_confusion_matrix(result.confusion_matrix, class_names, cm_path, title="Confusion Matrix (Validation)")
    logger.info(f"[WaferCNN] Confusion matrix saved to {cm_path}")
