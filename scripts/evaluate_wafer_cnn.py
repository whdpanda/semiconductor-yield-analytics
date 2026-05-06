"""Evaluate a trained WaferCNN on the WM-811K test split.

Usage:
    python scripts/evaluate_wafer_cnn.py
    python scripts/evaluate_wafer_cnn.py --checkpoint outputs/models/wafer_cnn_best.pth
    python scripts/evaluate_wafer_cnn.py --run-id run_20240101_120000
    python scripts/evaluate_wafer_cnn.py --max-samples 5000

Requires:
    - data/raw/wm811k/LSWMD.pkl  (see README for download instructions)
    - outputs/models/wafer_cnn_best.pth  (produced by train_wafer_cnn.py)

Outputs (under outputs/reports/wafer/runs/<run_id>/):
    confusion_matrix_test.png
    confusion_matrix_test_normalized.png
    classification_report.csv / .json
    evaluation_summary.json              -- includes prediction_distribution, run_id
    evaluation_metrics.json              -- legacy per-class metrics
    calibration_report.csv              -- threshold sweep on val split
    calibration_summary.json            -- baseline vs. calibrated comparison
    confusion_matrix_calibrated.png     -- calibrated test predictions
    classification_report_calibrated.csv/.json

NOTE: This is a portfolio project using the public WM-811K dataset.
      Reported metrics do not represent real fab deployment performance.
      WM-811K has ~79% 'none' class; macro-F1 and per-class recall are
      the meaningful metrics. Accuracy alone is misleading.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")

from loguru import logger

from semiconductor_yield.config import (
    MODELS_DIR,
    WAFER_DEFECT_CLASSES,
    WAFER_REPORTS_DIR,
    WM811K_PKL,
)
from semiconductor_yield.wafer.data_loader import WM811KLoader
from semiconductor_yield.wafer.dataset import WaferMapDataset
from semiconductor_yield.wafer.evaluate import (
    collect_probabilities,
    evaluate_model,
    load_model,
    plot_confusion_matrix,
    plot_confusion_matrix_normalized,
    save_calibration_report,
    save_classification_report,
    save_evaluation_summary,
    save_misclassified,
)
from semiconductor_yield.wafer.preprocess import stratified_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained WaferCNN on the WM-811K test split"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODELS_DIR / "wafer_cnn_best.pth",
        help="Path to .pth checkpoint (default: outputs/models/wafer_cnn_best.pth)",
    )
    parser.add_argument("--data", type=Path, default=WM811K_PKL,
                        help="Path to LSWMD.pkl (default: data/raw/wm811k/LSWMD.pkl)")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cpu', or 'cuda' (default: auto)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit total labeled samples (useful for quick smoke runs)")
    parser.add_argument("--misclassified", type=int, default=16,
                        help="Number of misclassified examples to plot (default: 16)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run identifier for output directory. Auto-generated if not set.")
    args = parser.parse_args(argv)

    # ── Guards ─────────────────────────────────────────────────────────────────
    if not args.data.exists():
        print(
            f"\nWM-811K data not found at: {args.data}\n\n"
            "To download:\n"
            "  1. Visit https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map\n"
            "  2. Download LSWMD.pkl (~350 MB)\n"
            "  3. Place it at data/raw/wm811k/LSWMD.pkl\n"
        )
        return 1

    if not args.checkpoint.exists():
        print(
            f"\nCheckpoint not found at: {args.checkpoint}\n\n"
            "Train the model first:\n"
            "  python scripts/train_wafer_cnn.py\n"
        )
        return 1

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info(f"Loading WM-811K from {args.data} ...")
    loader = WM811KLoader(pkl_path=args.data)
    samples = loader.load(labeled_only=True)
    logger.info(f"Loaded {len(samples):,} labeled samples")

    if args.max_samples and len(samples) > args.max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, args.max_samples)
        logger.info(f"Subsampled to {len(samples):,} samples (--max-samples)")

    # ── Reproduce training split to isolate test set ───────────────────────────
    splits = stratified_split(samples)
    logger.info(
        f"Split: train={len(splits.train):,}  val={len(splits.val):,}  "
        f"test={len(splits.test):,}"
    )

    # ── Device ─────────────────────────────────────────────────────────────────
    import torch
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
    class_names = list(WAFER_DEFECT_CLASSES)
    num_classes = len(class_names)
    model = load_model(args.checkpoint, num_classes=num_classes, dropout=0.0, device=device)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Build val and test DataLoaders ────────────────────────────────────────
    from torch.utils.data import DataLoader

    val_ds   = WaferMapDataset(splits.val,  augment=False)
    test_ds  = WaferMapDataset(splits.test, augment=False)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── Evaluate on test split ─────────────────────────────────────────────────
    print(f"\nEvaluating on {len(splits.test):,} test samples ...")
    result = evaluate_model(model, test_loader, device, class_names)

    print(f"\n  Accuracy          : {result.accuracy:.4f}")
    print(f"  Macro F1          : {result.macro_f1:.4f}")
    print(f"  Macro Precision   : {result.macro_precision:.4f}")
    print(f"  Macro Recall      : {result.macro_recall:.4f}")
    print(f"  Weighted F1       : {result.weighted_f1:.4f}")
    print()
    print("  Per-class metrics:")
    for name, m in result.per_class.items():
        print(
            f"    {name:<12}  f1={m['f1']:.4f}  precision={m['precision']:.4f}"
            f"  recall={m['recall']:.4f}  support={m['support']}"
        )

    # ── Save reports ───────────────────────────────────────────────────────────
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    report_dir = Path(WAFER_REPORTS_DIR) / "runs" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run_id={run_id}  reports -> {report_dir}")

    cm_path = report_dir / "confusion_matrix_test.png"
    plot_confusion_matrix(
        result.confusion_matrix, class_names, cm_path,
        title="Confusion Matrix (Test Split)",
    )
    logger.info(f"Saved confusion matrix to {cm_path}")

    cm_norm_path = report_dir / "confusion_matrix_test_normalized.png"
    plot_confusion_matrix_normalized(
        result.confusion_matrix, class_names, cm_norm_path,
        title="Confusion Matrix Normalized (Test Split)",
    )
    logger.info(f"Saved normalized confusion matrix to {cm_norm_path}")

    misclass_path = report_dir / "misclassified.png"
    save_misclassified(
        model, test_ds, device, misclass_path,
        n_examples=args.misclassified, class_names=class_names,
    )
    logger.info(f"Saved misclassified examples to {misclass_path}")

    save_classification_report(result, report_dir)
    logger.info(f"Saved classification report to {report_dir}")

    save_evaluation_summary(
        result, report_dir, split_name="test",
        n_samples=len(splits.test), run_id=run_id,
    )
    logger.info(f"Saved evaluation summary to {report_dir}")

    metrics = {
        "disclaimer": (
            "Metrics on WM-811K public dataset (portfolio project -- "
            "not real fab production performance)."
        ),
        "checkpoint": str(args.checkpoint),
        "n_test":     len(splits.test),
        "accuracy":   round(result.accuracy, 4),
        "macro_f1":   round(result.macro_f1, 4),
        "per_class":  result.per_class,
    }
    metrics_path = report_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # ── Threshold calibration ──────────────────────────────────────────────────
    print(f"\nRunning threshold calibration on {len(splits.val):,} val samples ...")
    probs_val,  y_val  = collect_probabilities(model, val_loader,  device)
    probs_test, y_test = collect_probabilities(model, test_loader, device)

    cal_result = save_calibration_report(
        probs_cal=probs_val,
        y_true_cal=y_val,
        probs_eval=probs_test,
        y_true_eval=y_test,
        class_names=class_names,
        output_dir=report_dir,
        thresholds=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    logger.info(f"Calibration complete — recommended threshold: {cal_result['recommended_threshold']}")

    rec_t = cal_result["recommended_threshold"]
    print(f"\n  Threshold calibration (recommended threshold = {rec_t}):")
    print(f"    none_recall       : {cal_result['baseline_none_recall']:.4f}  →  {cal_result['calibrated_none_recall']:.4f}")
    print(f"    false_alarm_rate  : {cal_result['baseline_false_alarm_rate']:.4f}  →  {cal_result['calibrated_false_alarm_rate']:.4f}")
    print(f"    scratch_precision : {cal_result['baseline_scratch_precision']:.4f}  →  {cal_result['calibrated_scratch_precision']:.4f}")
    print(f"    defect_recall     : {cal_result['baseline_defect_recall']:.4f}  →  {cal_result['calibrated_defect_recall']:.4f}")
    print(f"    macro_f1          : {cal_result['baseline_macro_f1']:.4f}  →  {cal_result['calibrated_macro_f1']:.4f}")

    print(f"\nrun_id: {run_id}")
    print(f"Outputs saved to {report_dir}/")
    print(f"  confusion_matrix_test.png")
    print(f"  confusion_matrix_test_normalized.png")
    print(f"  classification_report.csv")
    print(f"  classification_report.json")
    print(f"  evaluation_summary.json")
    print(f"  misclassified.png")
    print(f"  evaluation_metrics.json")
    print(f"  calibration_report.csv               (threshold sweep on val)")
    print(f"  calibration_summary.json             (baseline vs. calibrated)")
    print(f"  confusion_matrix_calibrated.png      (threshold={rec_t})")
    print(f"  classification_report_calibrated.csv (threshold={rec_t})")
    print(f"  classification_report_calibrated.json")
    print()
    print("NOTE: WM-811K has ~79% 'none' class -- macro-F1 and per-class recall")
    print("      are the meaningful metrics; accuracy alone is misleading.")
    print("NOTE: This is a portfolio project using the public WM-811K dataset.")
    print("      These metrics do not represent real fab deployment performance.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
