"""Train the WaferCNN baseline on WM-811K wafer map data.

Usage:
    python scripts/train_wafer_cnn.py
    python scripts/train_wafer_cnn.py --epochs 50 --batch-size 128
    python scripts/train_wafer_cnn.py --balanced-subset --samples-per-class 600
    python scripts/train_wafer_cnn.py --class-weight-mode sqrt_inverse
    python scripts/train_wafer_cnn.py --no-weighted-sampler

Requires data/raw/wm811k/LSWMD.pkl -- download from Kaggle:
    https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

Outputs (under outputs/reports/wafer/runs/<run_id>/):
    training_metrics.json
    confusion_matrix_val.png
    confusion_matrix_val_normalized.png
    classification_report.csv / .json
    evaluation_summary.json         -- includes prediction_distribution

outputs/models/wafer_cnn_best.pth   -- best checkpoint by val macro F1

NOTE: This is a portfolio project using the public WM-811K dataset.
      Reported metrics do not represent real fab deployment performance.
      WM-811K has ~79% 'none' class; macro-F1 and per-class recall are
      the meaningful metrics. Accuracy alone is misleading.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")

from loguru import logger

from semiconductor_yield.config import WAFER_DEFECT_CLASSES, WM811K_PKL
from semiconductor_yield.wafer.data_loader import WM811KLoader
from semiconductor_yield.wafer.dataset import WaferMapDataset
from semiconductor_yield.wafer.preprocess import stratified_split
from semiconductor_yield.wafer.train import fit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train WaferCNN on WM-811K wafer map dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",       type=Path, default=WM811K_PKL,
                        help="Path to LSWMD.pkl")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--device",     default="auto",
                        help="'auto', 'cpu', or 'cuda'")
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "inverse", "sqrt_inverse"],
        default="sqrt_inverse",
        help=(
            "Sampling weight strategy for WeightedRandomSampler. "
            "'sqrt_inverse' (default) is gentler and avoids single-class collapse. "
            "'inverse' is aggressive and can cause collapse on WM-811K. "
            "Ignored when --balanced-subset is set."
        ),
    )
    parser.add_argument(
        "--balanced-subset",
        action="store_true",
        help=(
            "Sub-sample at most --samples-per-class examples per class and "
            "train with shuffle=True (no WeightedRandomSampler needed). "
            "Recommended for debugging or when collapse is observed."
        ),
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=500,
        help="Max samples per class when --balanced-subset is active.",
    )
    parser.add_argument("--no-weighted-sampler", action="store_true",
                        help="Disable WeightedRandomSampler entirely (train on raw distribution).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit total labeled samples (for quick smoke runs).")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run identifier for output directory. Auto-generated if not set.")
    args = parser.parse_args(argv)

    # ── Data guard ─────────────────────────────────────────────────────────────
    if not args.data.exists():
        print(
            f"\nWM-811K data not found at: {args.data}\n\n"
            "To download:\n"
            "  1. Visit https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map\n"
            "  2. Download LSWMD.pkl (~350 MB)\n"
            "  3. Place it at data/raw/wm811k/LSWMD.pkl\n"
        )
        return 1

    # ── Load ───────────────────────────────────────────────────────────────────
    logger.info(f"Loading WM-811K from {args.data} ...")
    loader = WM811KLoader(pkl_path=args.data)
    samples = loader.load(labeled_only=True)
    logger.info(f"Loaded {len(samples):,} labeled samples")

    if args.max_samples and len(samples) > args.max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, args.max_samples)
        logger.info(f"Subsampled to {len(samples):,} samples (--max-samples)")

    # ── Split ──────────────────────────────────────────────────────────────────
    splits = stratified_split(samples)
    logger.info(
        f"Split: train={len(splits.train):,}  val={len(splits.val):,}  test={len(splits.test):,}"
    )

    train_ds = WaferMapDataset(splits.train, augment=True)
    val_ds   = WaferMapDataset(splits.val,   augment=False)

    # ── Train ──────────────────────────────────────────────────────────────────
    use_sampler = not args.no_weighted_sampler and not args.balanced_subset

    print(f"\nTraining WaferCNN on {len(splits.train):,} samples")
    print(f"  Epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  device={args.device}")
    if args.balanced_subset:
        print(f"  Mode: balanced subset ({args.samples_per_class} samples/class)")
    elif use_sampler:
        print(f"  Mode: WeightedRandomSampler (class-weight-mode={args.class_weight_mode})")
    else:
        print(f"  Mode: no sampler (raw class distribution)")
    print(f"\nNOTE: WM-811K has ~79% 'none' class -- macro-F1 is the key metric.")
    print(f"NOTE: This is a portfolio project. Metrics are on public WM-811K data only.\n")

    metrics = fit(
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        use_weighted_sampler=use_sampler,
        class_weight_mode=args.class_weight_mode,
        balanced_subset=args.balanced_subset,
        samples_per_class=args.samples_per_class,
        device_str=args.device,
        class_names=list(WAFER_DEFECT_CLASSES),
        run_id=args.run_id,
    )

    run_id = metrics["run_id"]
    run_dir = f"outputs/reports/wafer/runs/{run_id}"
    print(f"\nTraining complete.  run_id={run_id}")
    print(f"  Best val macro F1 = {metrics['best_val_macro_f1']:.4f}  (epoch {metrics['best_epoch']})")
    print(f"  Checkpoint  : outputs/models/wafer_cnn_best.pth")
    print(f"  Reports dir : {run_dir}/")
    print(f"    training_metrics.json")
    print(f"    confusion_matrix_val.png")
    print(f"    confusion_matrix_val_normalized.png")
    print(f"    classification_report.csv / .json")
    print(f"    evaluation_summary.json")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
