"""Evaluation utilities for WaferCNN (Module A).

Public API:
  evaluate_model(model, loader, device, class_names)          -> EvalResult
  collect_probabilities(model, loader, device)                -> (probs, y_true)
  apply_none_bias_threshold(probs, none_idx, threshold)       -> y_pred
  save_calibration_report(probs_cal, y_cal, probs_eval, y_eval, class_names, output_dir) -> dict
  plot_confusion_matrix(cm, class_names, output_path)
  plot_confusion_matrix_normalized(cm, class_names, output_path)
  save_classification_report(result, output_dir)
  save_evaluation_summary(result, output_dir, split_name, n_samples)
  save_misclassified(model, dataset, device, output_dir, n_examples, class_names)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
    macro_precision: float
    macro_recall: float
    weighted_f1: float
    weighted_precision: float
    weighted_recall: float
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
        EvalResult with accuracy, macro/weighted F1, per-class metrics, and confusion matrix.
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

    macro_avg    = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return EvalResult(
        y_true=y_true,
        y_pred=y_pred,
        accuracy=accuracy,
        macro_f1=macro_f1,
        macro_precision=round(float(macro_avg.get("precision", 0.0)), 4),
        macro_recall=round(float(macro_avg.get("recall", 0.0)), 4),
        weighted_f1=round(float(weighted_avg.get("f1-score", 0.0)), 4),
        weighted_precision=round(float(weighted_avg.get("precision", 0.0)), 4),
        weighted_recall=round(float(weighted_avg.get("recall", 0.0)), 4),
        per_class=per_class,
        confusion_matrix=cm,
        class_names=class_names,
    )


# ── Probability collection ────────────────────────────────────────────────────

def collect_probabilities(
    model: WaferCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on loader and return softmax probabilities and true labels.

    Args:
        model: Trained WaferCNN in eval mode.
        loader: DataLoader over the split.
        device: Torch device.

    Returns:
        Tuple of:
          - probs: shape (N, n_classes), softmax probabilities.
          - y_true: shape (N,), integer class labels.
    """
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(y.numpy().tolist())

    return np.vstack(all_probs), np.array(all_labels)


# ── Threshold calibration ──────────────────────────────────────────────────────

def apply_none_bias_threshold(
    probs: np.ndarray,
    none_idx: int,
    threshold: float,
) -> np.ndarray:
    """Apply a confidence threshold to suppress low-confidence defect predictions.

    For each sample, if the best non-none class probability is below
    ``threshold``, the prediction is overridden to the none class.  At
    ``threshold=0.0`` the result is identical to argmax (no change).

    This reduces false alarms (true-none samples predicted as defect) at the
    cost of lower defect recall.

    Args:
        probs: (N, n_classes) softmax probability array.
        none_idx: Index of the "none" (normal wafer) class.
        threshold: Minimum confidence required to predict any defect class.

    Returns:
        Integer prediction array of shape (N,).
    """
    n_classes = probs.shape[1]
    defect_cols = [i for i in range(n_classes) if i != none_idx]
    best_defect_prob = probs[:, defect_cols].max(axis=1)

    preds = probs.argmax(axis=1).copy()
    preds[best_defect_prob < threshold] = none_idx
    return preds


def _calibration_row_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    none_idx: int,
    scratch_idx: int,
    class_names: list[str],
) -> dict:
    """Compute one row of metrics for calibration_report.csv."""
    n_classes = len(class_names)
    acc = float((y_true == y_pred).mean())
    mac_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    wgt_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true, y_pred,
        labels=list(range(n_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    none_name    = class_names[none_idx]
    scratch_name = class_names[scratch_idx]
    none_recall      = float(report.get(none_name,    {}).get("recall",    0.0))
    scratch_prec     = float(report.get(scratch_name, {}).get("precision", 0.0))
    scratch_rec      = float(report.get(scratch_name, {}).get("recall",    0.0))

    true_none_mask   = (y_true == none_idx)
    n_true_none      = int(true_none_mask.sum())
    false_alarm_rate = float(((y_pred != none_idx) & true_none_mask).sum()) / max(n_true_none, 1)

    true_defect_mask = (y_true != none_idx)
    n_true_defect    = int(true_defect_mask.sum())
    defect_recall    = float(((y_pred != none_idx) & true_defect_mask).sum()) / max(n_true_defect, 1)

    return {
        "accuracy":          round(acc, 4),
        "macro_f1":          round(mac_f1, 4),
        "weighted_f1":       round(wgt_f1, 4),
        "none_recall":       round(none_recall, 4),
        "scratch_precision": round(scratch_prec, 4),
        "scratch_recall":    round(scratch_rec, 4),
        "false_alarm_rate":  round(false_alarm_rate, 4),
        "defect_recall":     round(defect_recall, 4),
    }


def compute_fab_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    none_idx: int | None = None,
    scratch_idx: int | None = None,
) -> dict:
    """Compute false-alarm-aware evaluation metrics for fab context.

    Returns a dict containing: accuracy, macro_f1, weighted_f1, none_recall,
    scratch_precision, scratch_recall, false_alarm_rate, defect_recall.

    Definitions:
      false_alarm_rate = count(true_none AND predicted_defect) / count(true_none)
      defect_recall    = count(true_defect AND predicted_defect) / count(true_defect)

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Ordered class name list (index = label).
        none_idx: Index of the none class. Inferred from class_names if None.
        scratch_idx: Index of the Scratch class. Inferred if None.

    Returns:
        Dict with 8 metric keys, all rounded to 4 decimal places.
    """
    if none_idx is None:
        none_idx = class_names.index("none") if "none" in class_names else len(class_names) - 1
    if scratch_idx is None:
        scratch_idx = (
            class_names.index("Scratch") if "Scratch" in class_names else 0
        )
    return _calibration_row_metrics(y_true, y_pred, none_idx, scratch_idx, class_names)


def save_calibration_report(
    probs_cal: np.ndarray,
    y_true_cal: np.ndarray,
    probs_eval: np.ndarray,
    y_true_eval: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    thresholds: list[float] | None = None,
    none_idx: int | None = None,
    scratch_idx: int | None = None,
) -> dict:
    """Sweep confidence thresholds and produce a false-alarm-aware calibration report.

    Threshold selection is done on the **calibration (val)** split;
    the final calibrated artefacts (confusion matrix, classification report)
    are computed on the **evaluation (test)** split using the recommended threshold.

    Outputs written to output_dir:
      - calibration_report.csv           (threshold sweep on val)
      - calibration_summary.json         (baseline vs. calibrated comparison)
      - confusion_matrix_calibrated.png  (calibrated test predictions)
      - classification_report_calibrated.csv/.json (calibrated test predictions)

    Args:
        probs_cal:   (N_val, n_classes) softmax probs — used to pick threshold.
        y_true_cal:  (N_val,) integer labels for val split.
        probs_eval:  (N_test, n_classes) softmax probs — used for final reporting.
        y_true_eval: (N_test,) integer labels for test split.
        class_names: Class name list, index-aligned.
        output_dir:  Directory to write all outputs.
        thresholds:  Thresholds to sweep. Defaults to [0.0, 0.3..0.9].
        none_idx:    Index of "none" class. Inferred from class_names if None.
        scratch_idx: Index of "Scratch" class. Inferred if None.

    Returns:
        Dict with keys: recommended_threshold, baseline_* and calibrated_* metrics.
    """
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if none_idx is None:
        none_idx = class_names.index("none") if "none" in class_names else len(class_names) - 1
    if scratch_idx is None:
        scratch_idx = class_names.index("Scratch") if "Scratch" in class_names else 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sweep thresholds on calibration (val) split ───────────────────────────
    cal_rows: list[dict] = []
    for t in thresholds:
        y_pred_t = apply_none_bias_threshold(probs_cal, none_idx, t)
        m = _calibration_row_metrics(y_true_cal, y_pred_t, none_idx, scratch_idx, class_names)
        cal_rows.append({"threshold": t, **m})

    # ── Save calibration_report.csv ───────────────────────────────────────────
    cal_report_path = output_dir / "calibration_report.csv"
    fieldnames = ["threshold", "accuracy", "macro_f1", "weighted_f1",
                  "none_recall", "scratch_precision", "scratch_recall",
                  "false_alarm_rate", "defect_recall"]
    with open(cal_report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cal_rows)
    logger.info(f"Calibration report saved → {cal_report_path}")

    # ── Select recommended threshold ──────────────────────────────────────────
    baseline_row = cal_rows[0]  # threshold=0.0 is identical to argmax
    baseline_none_recall   = baseline_row["none_recall"]
    baseline_false_alarm   = baseline_row["false_alarm_rate"]
    baseline_macro_f1      = baseline_row["macro_f1"]

    # Candidates: improve none_recall, keep macro_f1 and defect_recall reasonable
    candidates = [
        r for r in cal_rows
        if r["threshold"] > 0.0
        and r["none_recall"]  >= baseline_none_recall + 0.05
        and r["macro_f1"]     >= baseline_macro_f1 * 0.60
        and r["defect_recall"] >= 0.20
    ]
    if not candidates:
        # Relax: just require macro_f1 not catastrophic
        candidates = [
            r for r in cal_rows
            if r["threshold"] > 0.0
            and r["none_recall"] >= baseline_none_recall + 0.05
            and r["macro_f1"] >= baseline_macro_f1 * 0.50
        ]
    if not candidates:
        # Fallback: pick threshold with highest none_recall improvement
        candidates = [r for r in cal_rows if r["threshold"] > 0.0]

    # Score: reward none_recall improvement and false alarm reduction equally
    def _score(r: dict) -> float:
        nr_gain = r["none_recall"] - baseline_none_recall
        fa_red  = baseline_false_alarm - r["false_alarm_rate"]
        return 0.5 * nr_gain + 0.5 * fa_red

    recommended_row = max(candidates, key=_score)
    recommended_threshold = float(recommended_row["threshold"])

    # ── Apply recommended threshold to evaluation (test) split ────────────────
    y_pred_cal_recommended = apply_none_bias_threshold(probs_eval, none_idx, recommended_threshold)
    calibrated_metrics = _calibration_row_metrics(
        y_true_eval, y_pred_cal_recommended, none_idx, scratch_idx, class_names
    )

    # Baseline metrics on test split (threshold=0.0 == argmax)
    y_pred_baseline = apply_none_bias_threshold(probs_eval, none_idx, 0.0)
    base_test_metrics = _calibration_row_metrics(
        y_true_eval, y_pred_baseline, none_idx, scratch_idx, class_names
    )

    # ── Save calibration_summary.json ─────────────────────────────────────────
    tradeoff_note = (
        f"Threshold {recommended_threshold} was selected on the validation split. "
        f"none_recall improved from {base_test_metrics['none_recall']:.4f} to "
        f"{calibrated_metrics['none_recall']:.4f} on the test split. "
        f"false_alarm_rate improved from {base_test_metrics['false_alarm_rate']:.4f} to "
        f"{calibrated_metrics['false_alarm_rate']:.4f}. "
        f"defect_recall changed from {base_test_metrics['defect_recall']:.4f} to "
        f"{calibrated_metrics['defect_recall']:.4f}. "
        "Calibration trades defect recall for higher none recall (fewer false alarms). "
        "In a real fab context, a false alarm incurs tool downtime cost; "
        "the recommended threshold reflects that false alarms are expensive."
    )

    summary = {
        "disclaimer": (
            "Threshold calibration on WM-811K public dataset (portfolio project — "
            "not real fab deployment performance)."
        ),
        "calibration_split": "val",
        "evaluation_split":  "test",
        "thresholds_swept":  thresholds,
        "recommended_threshold": recommended_threshold,
        "false_alarm_rate_definition": (
            "false_alarm_rate = count(true_none AND predicted_defect) / count(true_none)"
        ),
        "baseline_accuracy":          base_test_metrics["accuracy"],
        "baseline_macro_f1":          base_test_metrics["macro_f1"],
        "baseline_none_recall":       base_test_metrics["none_recall"],
        "baseline_false_alarm_rate":  base_test_metrics["false_alarm_rate"],
        "baseline_scratch_precision": base_test_metrics["scratch_precision"],
        "baseline_scratch_recall":    base_test_metrics["scratch_recall"],
        "baseline_defect_recall":     base_test_metrics["defect_recall"],
        "calibrated_accuracy":          calibrated_metrics["accuracy"],
        "calibrated_macro_f1":          calibrated_metrics["macro_f1"],
        "calibrated_none_recall":       calibrated_metrics["none_recall"],
        "calibrated_false_alarm_rate":  calibrated_metrics["false_alarm_rate"],
        "calibrated_scratch_precision": calibrated_metrics["scratch_precision"],
        "calibrated_scratch_recall":    calibrated_metrics["scratch_recall"],
        "calibrated_defect_recall":     calibrated_metrics["defect_recall"],
        "tradeoff_note": tradeoff_note,
    }

    summary_path = output_dir / "calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Calibration summary saved → {summary_path}")

    # ── Save confusion_matrix_calibrated.png ──────────────────────────────────
    cm_cal = confusion_matrix(
        y_true_eval, y_pred_cal_recommended,
        labels=list(range(len(class_names))),
    )
    plot_confusion_matrix_normalized(
        cm_cal, class_names,
        output_dir / "confusion_matrix_calibrated.png",
        title=f"Confusion Matrix Calibrated (threshold={recommended_threshold})",
    )
    logger.info(f"Calibrated confusion matrix saved → {output_dir / 'confusion_matrix_calibrated.png'}")

    # ── Save classification_report_calibrated.csv/.json ───────────────────────
    n_classes = len(class_names)
    cal_class_report = classification_report(
        y_true_eval, y_pred_cal_recommended,
        labels=list(range(n_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cal_rows_per_class = [
        {
            "class_name": name,
            "precision":  round(float(cal_class_report.get(name, {}).get("precision", 0.0)), 4),
            "recall":     round(float(cal_class_report.get(name, {}).get("recall", 0.0)), 4),
            "f1-score":   round(float(cal_class_report.get(name, {}).get("f1-score", 0.0)), 4),
            "support":    int(cal_class_report.get(name, {}).get("support", 0)),
        }
        for name in class_names
    ]

    csv_path = output_dir / "classification_report_calibrated.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_name", "precision", "recall", "f1-score", "support"]
        )
        writer.writeheader()
        writer.writerows(cal_rows_per_class)

    json_path = output_dir / "classification_report_calibrated.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cal_rows_per_class, f, indent=2)
    logger.info(f"Calibrated classification report saved → {output_dir}")

    return {
        "recommended_threshold": recommended_threshold,
        **{f"baseline_{k}": v for k, v in base_test_metrics.items()},
        **{f"calibrated_{k}": v for k, v in calibrated_metrics.items()},
    }


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


# ── Normalized confusion matrix ───────────────────────────────────────────────

def plot_confusion_matrix_normalized(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion Matrix (Normalized)",
) -> None:
    """Save a row-normalized confusion matrix as PNG (cell values = per-class recall).

    Args:
        cm: Integer matrix of shape (n_classes, n_classes).
        class_names: Class labels for axes.
        output_path: Destination file path (.png).
        title: Figure title.
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Row-normalize: cm_norm[i,j] = fraction of true-class-i predicted as class-j
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            ax.text(
                j, i, f"{val:.0%}",
                ha="center", va="center", fontsize=7,
                color="white" if val > 0.5 else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Report savers ──────────────────────────────────────────────────────────────

def save_classification_report(result: EvalResult, output_dir: Path) -> None:
    """Write classification_report.csv and classification_report.json to output_dir.

    CSV columns: class_name, precision, recall, f1-score, support.

    Args:
        result: EvalResult from evaluate_model().
        output_dir: Directory to write reports into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "class_name": name,
            "precision":  result.per_class.get(name, {}).get("precision", 0.0),
            "recall":     result.per_class.get(name, {}).get("recall", 0.0),
            "f1-score":   result.per_class.get(name, {}).get("f1", 0.0),
            "support":    result.per_class.get(name, {}).get("support", 0),
        }
        for name in result.class_names
    ]

    csv_path = output_dir / "classification_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "classification_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def save_evaluation_summary(
    result: EvalResult,
    output_dir: Path,
    split_name: str = "val",
    n_samples: int | None = None,
    run_id: str | None = None,
    n_train: int = 0,
) -> None:
    """Write evaluation_summary.json to output_dir.

    Also logs a WARNING if any single class accounts for >80% of predictions,
    which indicates possible single-class collapse.

    Args:
        result: EvalResult from evaluate_model().
        output_dir: Directory to write summary into.
        split_name: "val" or "test".
        n_samples: Evaluation set size. Defaults to len(result.y_true).
        run_id: Training run identifier for cross-file traceability.
        n_train: Number of training samples; recorded for reproducibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_samples is None:
        n_samples = int(len(result.y_true))

    n_classes = len(result.class_names)
    n_pred = max(len(result.y_pred), 1)
    pred_counts = np.bincount(result.y_pred, minlength=n_classes)
    true_counts = np.bincount(result.y_true, minlength=n_classes)

    prediction_distribution = {
        result.class_names[i]: round(float(pred_counts[i] / n_pred), 4)
        for i in range(n_classes)
    }
    true_distribution = {
        result.class_names[i]: round(float(true_counts[i] / n_pred), 4)
        for i in range(n_classes)
    }

    # Collapse detection
    max_class = max(prediction_distribution, key=prediction_distribution.get)
    max_frac = prediction_distribution[max_class]
    if max_frac > 0.80:
        logger.warning(
            f"[collapse] '{max_class}' = {max_frac:.1%} of {split_name} predictions "
            f"(run_id={run_id}) -- possible single-class collapse. "
            "Check WeightedRandomSampler mode and loss."
        )

    summary = {
        "disclaimer": (
            "Metrics on WM-811K public dataset (portfolio project -- "
            "not real fab production performance)."
        ),
        "run_id":                  run_id,
        "split":                   split_name,
        "n_train":                 n_train,
        "num_validation_samples":  n_samples,
        "class_names":             result.class_names,
        "accuracy":                round(result.accuracy, 4),
        "macro_precision":         result.macro_precision,
        "macro_recall":            result.macro_recall,
        "macro_f1":                round(result.macro_f1, 4),
        "weighted_precision":      result.weighted_precision,
        "weighted_recall":         result.weighted_recall,
        "weighted_f1":             result.weighted_f1,
        "prediction_distribution": prediction_distribution,
        "true_distribution":       true_distribution,
    }

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


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
