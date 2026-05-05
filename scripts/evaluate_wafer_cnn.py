"""Evaluate a trained WaferCNN on the WM-811K test split.

Usage:
    python scripts/evaluate_wafer_cnn.py
    python scripts/evaluate_wafer_cnn.py --checkpoint outputs/models/wafer_cnn_best.pth
    python scripts/evaluate_wafer_cnn.py --max-samples 5000

Requires:
    - data/raw/wm811k/LSWMD.pkl  (see README for download instructions)
    - outputs/models/wafer_cnn_best.pth  (produced by train_wafer_cnn.py)

Outputs:
    outputs/reports/wafer/evaluation_metrics.json    -- per-class metrics on test split
    outputs/reports/wafer/confusion_matrix_test.png  -- confusion matrix on test split
    outputs/reports/wafer/misclassified.png          -- grid of misclassified examples

NOTE: This is a portfolio project using the public WM-811K dataset.
      Reported metrics do not represent real fab deployment performance.
"""

import argparse
import json
import sys
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
    evaluate_model,
    load_model,
    plot_confusion_matrix,
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

    # ── Evaluate on test split ─────────────────────────────────────────────────
    from torch.utils.data import DataLoader

    test_ds = WaferMapDataset(splits.test, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\nEvaluating on {len(splits.test):,} test samples ...")
    result = evaluate_model(model, test_loader, device, class_names)

    print(f"\n  Accuracy  : {result.accuracy:.4f}")
    print(f"  Macro F1  : {result.macro_f1:.4f}")
    print()
    print("  Per-class metrics:")
    for name, m in result.per_class.items():
        print(
            f"    {name:<12}  f1={m['f1']:.4f}  precision={m['precision']:.4f}"
            f"  recall={m['recall']:.4f}  support={m['support']}"
        )

    # ── Save reports ───────────────────────────────────────────────────────────
    report_dir = Path(WAFER_REPORTS_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    cm_path = report_dir / "confusion_matrix_test.png"
    plot_confusion_matrix(
        result.confusion_matrix, class_names, cm_path,
        title="Confusion Matrix (Test Split)",
    )
    logger.info(f"Saved confusion matrix to {cm_path}")

    misclass_path = report_dir / "misclassified.png"
    save_misclassified(
        model, test_ds, device, misclass_path,
        n_examples=args.misclassified, class_names=class_names,
    )
    logger.info(f"Saved misclassified examples to {misclass_path}")

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

    print(f"\nOutputs saved to {report_dir}/")
    print(f"  confusion_matrix_test.png")
    print(f"  misclassified.png")
    print(f"  evaluation_metrics.json")
    print()
    print("NOTE: This is a portfolio project using the public WM-811K dataset.")
    print("      These metrics do not represent real fab deployment performance.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
