"""Evaluate trained anomaly detectors on the held-out test split.

Usage:
    python scripts/evaluate_process_anomaly.py

Requires trained models and split_info.json in outputs/models/
(run train_process_anomaly.py first).

Outputs (written to outputs/reports/process/):
  anomaly_scores.csv      — per-sample scores and binary predictions (test split)
  anomaly_summary.json    — precision / recall / F1 per model (test split)
  feature_importance.csv  — feature contribution from IF and AE

IMPORTANT: All metrics are computed on SIMULATED data with injected anomalies.
           These numbers do not represent real fab performance.
           Evaluation is restricted to the held-out TEST split defined in
           split_info.json — training data is never included in metric computation.
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from semiconductor_yield.config import MODELS_DIR, PROCESS_REPORTS_DIR, SYNTHETIC_DIR
from semiconductor_yield.process.anomaly import AutoencoderDetector, IsolationForestDetector
from semiconductor_yield.process.features import get_feature_matrix


def _eval_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    n_total = len(y_true)
    n_anomaly = int(y_true.sum())
    n_normal = n_total - n_anomaly
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    far = round(fp / n_normal, 4) if n_normal > 0 else 0.0
    try:
        auc = round(float(roc_auc_score(y_true, scores)), 4)
    except Exception:
        auc = None
    return {
        "model": name,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": auc,
        "false_alarm_rate": far,
        "n_flagged": int(y_pred.sum()),
        "n_anomalies_gt": n_anomaly,
        "n_samples": n_total,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate process anomaly detectors (test split)")
    parser.add_argument("--data",       type=Path, default=SYNTHETIC_DIR / "process_data.csv")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESS_REPORTS_DIR)
    args = parser.parse_args(argv)

    # ── Guards ─────────────────────────────────────────────────────────────────
    if not args.data.exists():
        print(f"\nData not found: {args.data}\nRun: python scripts/generate_synthetic_process_data.py\n")
        return 1

    if_path    = args.models_dir / "isolation_forest.joblib"
    ae_path    = args.models_dir / "autoencoder.joblib"
    split_path = args.models_dir / "split_info.json"

    for p in [if_path, ae_path]:
        if not p.exists():
            print(f"\nModel not found: {p}\nRun: python scripts/train_process_anomaly.py\n")
            return 1

    if not split_path.exists():
        print(
            f"\nsplit_info.json not found: {split_path}\n"
            "Re-run train_process_anomaly.py to generate the lot-based split.\n"
        )
        return 1

    # ── Load split metadata and filter to test lots ────────────────────────────
    with open(split_path) as f:
        split_info = json.load(f)

    test_lots = set(split_info["test_lots"])
    n_test_lots = len(test_lots)

    logger.info(f"Loading data from {args.data}")
    df_full = pd.read_csv(args.data, parse_dates=["timestamp"])

    df = df_full[df_full["lot_id"].isin(test_lots)].reset_index(drop=True)

    if len(df) == 0:
        print(f"\nNo rows found for {n_test_lots} test lots. Check that --data matches the training data.\n")
        return 1

    logger.info(
        f"Test split: {n_test_lots} lots, {len(df):,} rows "
        f"({n_test_lots / split_info['n_lots_total']:.0%} of all lots)"
    )

    y_true = df["anomaly_label"].astype(int).values

    # ── Load models ────────────────────────────────────────────────────────────
    logger.info("Loading models ...")
    if_det = IsolationForestDetector.load(if_path)
    ae_det = AutoencoderDetector.load(ae_path)

    # Feature columns come from the trained model — matches the training feature set.
    X, feature_cols = get_feature_matrix(df, if_det._feature_cols)

    # ── Score ──────────────────────────────────────────────────────────────────
    scores_if = if_det.anomaly_scores(X)
    scores_ae = ae_det.anomaly_scores(X)
    pred_if   = if_det.predict(X)
    pred_ae   = ae_det.predict(X)

    pred_any = ((pred_if == 1) | (pred_ae == 1)).astype(int)
    pred_all = ((pred_if == 1) & (pred_ae == 1)).astype(int)
    scores_ensemble = (scores_if / (scores_if.max() + 1e-9) + scores_ae / (scores_ae.max() + 1e-9)) / 2

    # ── Metrics ────────────────────────────────────────────────────────────────
    results = [
        _eval_metrics("isolation_forest", y_true, pred_if,  scores_if),
        _eval_metrics("autoencoder",      y_true, pred_ae,  scores_ae),
        _eval_metrics("ensemble_any",     y_true, pred_any, scores_ensemble),
        _eval_metrics("ensemble_all",     y_true, pred_all, scores_ensemble),
    ]

    # ── Save anomaly_scores.csv ────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores_df = df[["lot_id", "wafer_id", "process_step", "timestamp", "anomaly_label", "anomaly_type"]].copy()
    scores_df["if_score"]     = scores_if
    scores_df["ae_score"]     = scores_ae
    scores_df["if_pred"]      = pred_if
    scores_df["ae_pred"]      = pred_ae
    scores_df["ensemble_any"] = pred_any
    scores_df["ensemble_all"] = pred_all
    scores_path = args.output_dir / "anomaly_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    logger.info(f"Scores saved → {scores_path}")

    # ── Save anomaly_summary.json ──────────────────────────────────────────────
    summary = {
        "_disclaimer": (
            "All metrics computed on SIMULATED data with controlled anomaly injection. "
            "These numbers do not represent real fab detection performance."
        ),
        "data_file": str(args.data),
        "split": {
            "strategy": split_info["split_strategy"],
            "eval_split": "test",
            "n_test_lots": n_test_lots,
            "n_test_rows": len(df),
            "n_lots_total": split_info["n_lots_total"],
        },
        "features_used": feature_cols,
        "models": {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in results},
    }
    summary_path = args.output_dir / "anomaly_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved → {summary_path}")

    # ── Save feature_importance.csv ────────────────────────────────────────────
    if_imp  = if_det.feature_importances()
    ae_rerr = ae_det.per_feature_reconstruction_error(X)

    feat_df = pd.DataFrame({
        "feature":                  feature_cols,
        "if_importance":            np.round(if_imp, 6),
        "ae_reconstruction_error":  np.round(ae_rerr, 8),
    }).sort_values("if_importance", ascending=False)
    feat_path = args.output_dir / "feature_importance.csv"
    feat_df.to_csv(feat_path, index=False)
    logger.info(f"Feature importance saved → {feat_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  Anomaly Detection Evaluation  (simulated data — test split)")
    print(f"{'=' * 64}")
    print(
        f"  Test split : {n_test_lots} lots, {len(df):,} rows  "
        f"({n_test_lots}/{split_info['n_lots_total']} lots held out)"
    )
    print(f"  Ground-truth anomalies: {int(y_true.sum())} ({y_true.mean():.1%})")
    print(f"\n  {'Model':<22} {'Precision':>9} {'Recall':>8} {'F1':>7} {'FAR':>7} {'ROC-AUC':>8}")
    print(f"  {'-'*22}  {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")
    for r in results:
        auc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "   N/A"
        print(
            f"  {r['model']:<22}  {r['precision']:>9.4f} {r['recall']:>8.4f} "
            f"{r['f1']:>7.4f} {r['false_alarm_rate']:>7.4f} {auc_str:>8}"
        )
    print(f"\n  Outputs → {args.output_dir}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
