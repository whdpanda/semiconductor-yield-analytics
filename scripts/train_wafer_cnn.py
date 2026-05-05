"""Train the WaferCNN baseline on WM-811K wafer map data.

Usage:
    python scripts/train_wafer_cnn.py
    python scripts/train_wafer_cnn.py --epochs 50 --batch-size 128
    python scripts/train_wafer_cnn.py --no-weighted-sampler

Requires data/raw/wm811k/LSWMD.pkl — download from Kaggle:
    https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

Outputs:
    outputs/models/wafer_cnn_best.pth       — best checkpoint by val macro F1
    outputs/reports/wafer/training_metrics.json
    outputs/reports/wafer/confusion_matrix_val.png

NOTE: This is a portfolio project using the public WM-811K dataset.
      Reported metrics do not represent real fab deployment performance.
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
        description="Train WaferCNN on WM-811K wafer map dataset"
    )
    parser.add_argument("--data",       type=Path, default=WM811K_PKL,
                        help="Path to LSWMD.pkl (default: data/raw/wm811k/LSWMD.pkl)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--device",     default="auto",
                        help="'auto', 'cpu', or 'cuda' (default: auto)")
    parser.add_argument("--no-weighted-sampler", action="store_true",
                        help="Disable WeightedRandomSampler (not recommended for WM-811K)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit total labeled samples (useful for quick smoke runs)")
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
    print(f"\nTraining WaferCNN on {len(splits.train):,} samples")
    print(f"  Epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  device={args.device}")
    print(f"  WeightedSampler={'no' if args.no_weighted_sampler else 'yes'}  "
          f"(recommended 'yes' for WM-811K's extreme class imbalance)")
    print(f"\nNOTE: This is a portfolio project — metrics are on WM-811K public data only.\n")

    metrics = fit(
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        use_weighted_sampler=not args.no_weighted_sampler,
        device_str=args.device,
        class_names=list(WAFER_DEFECT_CLASSES),
    )

    print(f"\nTraining complete.")
    print(f"  Best val macro F1 = {metrics['best_val_macro_f1']:.4f}  (epoch {metrics['best_epoch']})")
    print(f"  Checkpoint:  outputs/models/wafer_cnn_best.pth")
    print(f"  Metrics:     outputs/reports/wafer/training_metrics.json")
    print(f"  Confusion:   outputs/reports/wafer/confusion_matrix_val.png")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
