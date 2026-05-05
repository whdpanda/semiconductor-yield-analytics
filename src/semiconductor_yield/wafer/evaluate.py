"""Evaluation utilities for WaferCNN (Module A).

Public API:
  evaluate_model(model, loader, device, class_names) -> EvalResult
  plot_confusion_matrix(cm, class_names, output_path)
  save_misclassified(model, dataset, device, output_dir, n_examples, class_names)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from semiconductor_yield.models.wafer_cnn import WaferCNN
from semiconductor_yield.wafer.dataset import WaferMapDataset


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]   # class_name → {precision, recall, f1, support}
    confusion_matrix: np.ndarray
    class_names: list[str]


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_model(
    model: WaferCNN,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> EvalResult:
    """Run inference on loader and compute per-class metrics.

    Args:
        model: Trained WaferCNN in eval mode.
        loader: DataLoader over the evaluation split.
        device: Torch device.
        class_names: Ordered list of class names matching class indices.

    Returns:
        EvalResult with accuracy, macro F1, per-class metrics, and confusion matrix.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics via sklearn
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": round(float(report[name]["precision"]), 4),
            "recall":    round(float(report[name]["recall"]), 4),
            "f1":        round(float(report[name]["f1-score"]), 4),
            "support":   int(report[name]["support"]),
        }
        for name in class_names
        if name in report
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return EvalResult(
        y_true=y_true,
        y_pred=y_pred,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=cm,
        class_names=class_names,
    )


# ── Confusion matrix plot ──────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a confusion matrix as a PNG using matplotlib.

    Args:
        cm: Integer matrix of shape (n_classes, n_classes).
        class_names: Class labels for axes.
        output_path: Destination file path (.png).
        title: Figure title.
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Misclassified sample visualization ────────────────────────────────────────

def save_misclassified(
    model: WaferCNN,
    dataset: WaferMapDataset,
    device: torch.device,
    output_path: Path,
    n_examples: int = 16,
    class_names: list[str] | None = None,
) -> None:
    """Plot a grid of misclassified wafer maps with true vs. predicted labels.

    Samples are drawn from the first misclassified examples found when
    iterating the dataset in order.

    Args:
        model: Trained WaferCNN in eval mode.
        dataset: WaferMapDataset (augment=False) to sample from.
        device: Torch device.
        output_path: Destination PNG path.
        n_examples: Number of misclassified examples to show (≤ n_examples² cap).
        class_names: Class label list; index → name. Falls back to str(index).
    """
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = [str(i) for i in range(20)]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    wrong: list[tuple[np.ndarray, int, int]] = []  # (wmap, true, pred)

    with torch.no_grad():
        for i in range(len(dataset)):
            tensor, label = dataset[i]
            logit = model(tensor.unsqueeze(0).to(device))
            pred = int(logit.argmax(dim=1).item())
            true = int(label.item())
            if pred != true:
                wmap = tensor.squeeze(0).numpy()  # (H, W)
                wrong.append((wmap, true, pred))
            if len(wrong) >= n_examples:
                break

    if not wrong:
        # Write a placeholder PNG if no errors found
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No misclassified examples found", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return

    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (wmap, true, pred) in zip(axes, wrong):
        ax.imshow(wmap, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        t_name = class_names[true]  if true  < len(class_names) else str(true)
        p_name = class_names[pred]  if pred  < len(class_names) else str(pred)
        ax.set_title(f"T:{t_name}\nP:{p_name}", fontsize=7, color="red")
        ax.axis("off")

    fig.suptitle("Misclassified Wafer Maps (T=True, P=Predicted)", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: Path,
    num_classes: int = 9,
    dropout: float = 0.0,
    device: torch.device | None = None,
) -> WaferCNN:
    """Load a WaferCNN from a saved state dict.

    Args:
        checkpoint_path: Path to the .pth file saved by train.fit().
        num_classes: Must match the saved model's architecture.
        dropout: Set to 0.0 for inference (no dropout at eval time).
        device: Target device. Defaults to CPU.

    Returns:
        WaferCNN in eval mode.
    """
    if device is None:
        device = torch.device("cpu")
    model = WaferCNN(num_classes=num_classes, dropout=dropout)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
