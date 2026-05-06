"""Train the WaferCNN baseline on WM-811K wafer map data.

Usage:
    # default (weighted sampler, sqrt_inverse)
    python scripts/train_wafer_cnn.py

    # hybrid sampling — v2 recommended config
    python scripts/train_wafer_cnn.py --sampling-mode hybrid --epochs 10

    # hybrid sampling — small/CPU-friendly version
    python scripts/train_wafer_cnn.py \\
        --sampling-mode hybrid \\
        --none-samples 1000 --major-class-samples 500 \\
        --minor-class-samples 300 --rare-class-samples 200 \\
        --epochs 5

    # balanced subset (original v1 debug mode)
    python scripts/train_wafer_cnn.py --sampling-mode balanced --samples-per-class 600

Requires data/raw/wm811k/LSWMD.pkl -- download from Kaggle:
    https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

Outputs (under outputs/reports/wafer/runs/<run_id>/):
    training_metrics.json
    evaluation_metrics.json      -- fab-aware metrics (none_recall, false_alarm_rate, …)
    prediction_distribution.json
    confusion_matrix_val.png
    confusion_matrix_val_normalized.png
    classification_report.csv / .json
    evaluation_summary.json

outputs/models/wafer_cnn_best.pth   -- best checkpoint by val macro F1

NOTE: This is a portfolio project using the public WM-811K dataset.
      Reported metrics do not represent real fab deployment performance.
      WM-811K has ~79% 'none' class; macro-F1 and per-class recall are
      the meaningful metrics. Accuracy alone is misleading.

NOTE: README should be updated after v2 training result is accepted.
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
    # ── Data and model ─────────────────────────────────────────────────────────
    parser.add_argument("--data",        type=Path,  default=WM811K_PKL,
                        help="Path to LSWMD.pkl")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--device",      default="auto",
                        help="'auto', 'cpu', or 'cuda'")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit total labeled samples (for quick smoke runs).")
    parser.add_argument("--run-id",      type=str, default=None,
                        help="Run identifier for output directory. Auto-generated if not set.")

    # ── Sampling mode ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--sampling-mode",
        choices=["hybrid", "balanced", "weighted", "none"],
        default="weighted",
        help=(
            "Sampling strategy:\n"
            "  hybrid   — class-group-aware subset (none>defect, see --*-samples);\n"
            "             recommended for v2 false-alarm reduction\n"
            "  balanced — uniform cap per class (see --samples-per-class)\n"
            "  weighted — WeightedRandomSampler on full data (see --class-weight-mode)\n"
            "  none     — raw class distribution, no rebalancing"
        ),
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "inverse", "sqrt_inverse"],
        default="sqrt_inverse",
        help=(
            "Weight mode for WeightedRandomSampler. "
            "Used with --sampling-mode weighted or hybrid. "
            "'sqrt_inverse' avoids single-class collapse."
        ),
    )

    # ── Hybrid mode ────────────────────────────────────────────────────────────
    parser.add_argument("--none-samples",         type=int, default=3000,
                        help="Max none-class samples (hybrid mode).")
    parser.add_argument("--major-class-samples",  type=int, default=1000,
                        help="Max samples per major defect class (Edge-Ring, Edge-Loc, Center, Loc).")
    parser.add_argument("--minor-class-samples",  type=int, default=500,
                        help="Max samples per minor defect class (Scratch, Random, Donut).")
    parser.add_argument("--rare-class-samples",   type=int, default=300,
                        help="Max samples per rare class (Near-full).")

    # ── Balanced mode ──────────────────────────────────────────────────────────
    parser.add_argument("--samples-per-class", type=int, default=500,
                        help="Max samples per class when --sampling-mode balanced.")

    # ── Legacy flags (kept for backward compatibility) ─────────────────────────
    parser.add_argument("--balanced-subset",    action="store_true",
                        help="Equivalent to --sampling-mode balanced (legacy flag).")
    parser.add_argument("--no-weighted-sampler",action="store_true",
                        help="Equivalent to --sampling-mode none (legacy flag).")

    args = parser.parse_args(argv)

    # Resolve legacy flags
    if args.balanced_subset:
        args.sampling_mode = "balanced"
    if args.no_weighted_sampler:
        args.sampling_mode = "none"

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

    # ── Map sampling-mode to fit() args ───────────────────────────────────────
    mode = args.sampling_mode
    hybrid_subset   = (mode == "hybrid")
    balanced_subset = (mode == "balanced")
    use_sampler     = (mode in ("weighted", "hybrid"))

    print(f"\nTraining WaferCNN on {len(splits.train):,} samples")
    print(f"  Epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  device={args.device}")
    print(f"  Sampling mode: {mode}")
    if mode == "hybrid":
        print(
            f"  Hybrid config: none={args.none_samples}  major={args.major_class_samples}"
            f"  minor={args.minor_class_samples}  rare={args.rare_class_samples}"
        )
        print(f"  Class-weight-mode (on hybrid subset): {args.class_weight_mode}")
    elif mode == "balanced":
        print(f"  Balanced: {args.samples_per_class} samples/class")
    elif mode == "weighted":
        print(f"  WeightedRandomSampler: {args.class_weight_mode}")
    print(f"\nNOTE: WM-811K has ~79% 'none' class -- macro-F1 is the key metric.")
    print(f"NOTE: This is a portfolio project. Metrics are on public WM-811K data only.\n")

    metrics = fit(
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        use_weighted_sampler=use_sampler,
        class_weight_mode=args.class_weight_mode,
        balanced_subset=balanced_subset,
        samples_per_class=args.samples_per_class,
        hybrid_subset=hybrid_subset,
        none_samples=args.none_samples,
        major_class_samples=args.major_class_samples,
        minor_class_samples=args.minor_class_samples,
        rare_class_samples=args.rare_class_samples,
        device_str=args.device,
        class_names=list(WAFER_DEFECT_CLASSES),
        run_id=args.run_id,
    )

    run_id  = metrics["run_id"]
    run_dir = f"outputs/reports/wafer/runs/{run_id}"
    print(f"\nTraining complete.  run_id={run_id}")
    print(f"  Best val macro F1 = {metrics['best_val_macro_f1']:.4f}  (epoch {metrics['best_epoch']})")
    print(f"  Checkpoint  : outputs/models/wafer_cnn_best.pth")
    print(f"  Reports dir : {run_dir}/")
    print(f"    training_metrics.json")
    print(f"    evaluation_metrics.json      (none_recall / false_alarm_rate / …)")
    print(f"    prediction_distribution.json")
    print(f"    confusion_matrix_val.png")
    print(f"    confusion_matrix_val_normalized.png")
    print(f"    classification_report.csv / .json")
    print(f"    evaluation_summary.json")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
