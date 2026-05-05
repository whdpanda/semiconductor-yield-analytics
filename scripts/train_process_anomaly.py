"""Train Isolation Forest and Autoencoder anomaly detectors.

Usage:
    python scripts/train_process_anomaly.py
    python scripts/train_process_anomaly.py --feature-set process_only
    python scripts/train_process_anomaly.py --epochs 100 --n-estimators 200

The dataset is split by lot_id (grouped) into train / val / test before any
model fitting. Detectors are trained on the train split only. The val and test
splits are never seen during training. Evaluation metrics are computed by
evaluate_process_anomaly.py using the held-out test split.

Split metadata is saved to split_info.json alongside the model files so that
evaluate_process_anomaly.py can reconstruct the identical test split.

Training is unsupervised — the anomaly_label column is never read.
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger

from semiconductor_yield.config import MODELS_DIR, SYNTHETIC_DIR
from semiconductor_yield.process.anomaly import AutoencoderDetector, IsolationForestDetector
from semiconductor_yield.process.features import FEATURE_SET_MAP, get_feature_matrix, lot_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train process anomaly detectors")
    parser.add_argument("--data", type=Path, default=SYNTHETIC_DIR / "process_data.csv")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Isolation Forest trees (default 200)")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Expected anomaly fraction for IF (default 0.05)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Autoencoder training epochs (default 60)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 16, 8],
                        help="Autoencoder encoder layer sizes (default 32 16 8)")
    parser.add_argument("--feature-set", choices=["process_only", "full"], default="full",
                        help="'process_only' (in-situ only) or 'full' (adds offline metrology). Default: full")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Fraction of lots for validation (default 0.15)")
    parser.add_argument("--test-frac", type=float, default=0.15,
                        help="Fraction of lots for held-out test (default 0.15)")
    args = parser.parse_args(argv)

    if not args.data.exists():
        print(
            f"\nData not found: {args.data}\n"
            "Generate it first:\n"
            "  python scripts/generate_synthetic_process_data.py\n"
        )
        return 1

    train_frac = round(1.0 - args.val_frac - args.test_frac, 10)
    if train_frac <= 0:
        print(f"\nval_frac ({args.val_frac}) + test_frac ({args.test_frac}) must be < 1.0\n")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and split by lot_id ───────────────────────────────────────────────
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df):,} rows, {df['lot_id'].nunique()} lots")

    df_train, df_val, df_test = lot_split(
        df,
        train_frac=train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    train_lots = sorted(df_train["lot_id"].unique().tolist())
    val_lots   = sorted(df_val["lot_id"].unique().tolist())
    test_lots  = sorted(df_test["lot_id"].unique().tolist())

    print(f"\nLot-based split  (seed=42, strategy=lot_id_grouped)")
    print(f"  Train : {len(train_lots):3d} lots,  {len(df_train):5,} rows")
    print(f"  Val   : {len(val_lots):3d} lots,  {len(df_val):5,} rows")
    print(f"  Test  : {len(test_lots):3d} lots,  {len(df_test):5,} rows  ← held out, not used during training")

    # ── Build feature matrix from train split only ─────────────────────────────
    feature_list = FEATURE_SET_MAP[args.feature_set]
    X_train, feature_cols = get_feature_matrix(df_train, feature_list)
    logger.info(f"Feature matrix (train): {X_train.shape[0]} × {X_train.shape[1]}  features={feature_cols}")

    print(f"\nFeature set: {args.feature_set!r}  ({len(feature_cols)} features)")
    print(f"Training on {X_train.shape[0]:,} samples × {X_train.shape[1]} features  (train split only)")
    print(f"Features: {feature_cols}")
    print(f"Note: anomaly_label column is NOT used during training (unsupervised)")

    # ── Isolation Forest ───────────────────────────────────────────────────────
    print(f"\n[1/2] Isolation Forest  (n_estimators={args.n_estimators}) ...")
    if_det = IsolationForestDetector(
        contamination=args.contamination,
        n_estimators=args.n_estimators,
    )
    if_det.fit(X_train, feature_cols)
    if_path = args.output_dir / "isolation_forest.joblib"
    if_det.save(if_path)

    scores_if = if_det.anomaly_scores(X_train)
    print(f"  Score range (train): [{scores_if.min():.4f}, {scores_if.max():.4f}]")
    print(f"  Threshold (95th pct): {if_det._threshold:.4f}")
    print(f"  Flagged as anomaly (train): {int((scores_if >= if_det._threshold).sum())} / {len(X_train)}")

    # ── Autoencoder ────────────────────────────────────────────────────────────
    print(f"\n[2/2] Autoencoder  (hidden_dims={args.hidden_dims}, epochs={args.epochs}) ...")
    ae_det = AutoencoderDetector(
        hidden_dims=tuple(args.hidden_dims),
        epochs=args.epochs,
    )
    ae_det.fit(X_train, feature_cols)
    ae_path = args.output_dir / "autoencoder.joblib"
    ae_det.save(ae_path)

    scores_ae = ae_det.anomaly_scores(X_train)
    print(f"  Score range (train): [{scores_ae.min():.6f}, {scores_ae.max():.6f}]")
    print(f"  Threshold (95th pct): {ae_det._threshold:.6f}")
    print(f"  Flagged as anomaly (train): {int((scores_ae >= ae_det._threshold).sum())} / {len(X_train)}")

    # ── Save split metadata ────────────────────────────────────────────────────
    split_info = {
        "split_strategy": "lot_id_grouped",
        "random_state": 42,
        "feature_set": args.feature_set,
        "n_lots_total": df["lot_id"].nunique(),
        "train_lots": train_lots,
        "val_lots": val_lots,
        "test_lots": test_lots,
        "n_rows": {
            "train": len(df_train),
            "val":   len(df_val),
            "test":  len(df_test),
        },
    }
    split_path = args.output_dir / "split_info.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nModels and split info saved:")
    print(f"  {if_path}")
    print(f"  {ae_path}")
    print(f"  {split_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
