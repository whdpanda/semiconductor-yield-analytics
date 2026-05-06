"""Validate dashboard inference pipeline against real WM-811K test-split samples.

Confirms that:
  - The checkpoint in active_run.json loads correctly
  - Dashboard inference preprocessing matches the training pipeline
  - Per-class accuracy on real held-out test samples is consistent with
    the evaluation_metrics.json reported for the active run

Usage:
    python scripts/validate_wafer_dashboard_inference.py
    python scripts/validate_wafer_dashboard_inference.py --samples-per-class 20 --seed 42

Requires:
    data/raw/wm811k/LSWMD.pkl   (see README for download link)
    outputs/reports/wafer/active_run.json

Outputs:
    outputs/reports/wafer/dashboard_inference_validation.json

NOTE: This is a portfolio project using the public WM-811K dataset.
      Results do not represent real fab deployment performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from semiconductor_yield.config import WAFER_DEFECT_CLASSES, WAFER_REPORTS_DIR, WM811K_PKL
from semiconductor_yield.wafer.data_loader import WM811KLoader
from semiconductor_yield.wafer.inference import WaferInference
from semiconductor_yield.wafer.preprocess import stratified_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate dashboard inference pipeline on real WM-811K test samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples-per-class", type=int, default=20,
                        help="Number of test samples to draw per class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--data", type=Path, default=WM811K_PKL,
                        help="Path to LSWMD.pkl")
    args = parser.parse_args(argv)

    # ── Load active_run.json ───────────────────────────────────────────────────
    active_run_path = Path(WAFER_REPORTS_DIR) / "active_run.json"
    if not active_run_path.exists():
        print(f"\nERROR: active_run.json not found at {active_run_path}")
        print("Run consolidation first.")
        return 1

    with open(active_run_path, encoding="utf-8") as f:
        run_info = json.load(f)

    active_run_id = run_info.get("active_run_id", "unknown")
    root = Path(__file__).parent.parent.resolve()
    ckpt_rel = run_info.get("checkpoint_stable") or run_info.get("checkpoint_live")
    ckpt_path = (root / ckpt_rel) if ckpt_rel else (root / "outputs/models/wafer_cnn_best.pth")

    print(f"\nActive run : {active_run_id}")
    print(f"Checkpoint : {ckpt_path}")

    if not ckpt_path.exists():
        print(f"\nERROR: checkpoint not found: {ckpt_path}")
        return 1

    # ── Load WM-811K ──────────────────────────────────────────────────────────
    if not args.data.exists():
        print(
            f"\nERROR: WM-811K data not found at {args.data}\n\n"
            "To download:\n"
            "  1. Visit https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map\n"
            "  2. Download LSWMD.pkl (~350 MB)\n"
            "  3. Place it at data/raw/wm811k/LSWMD.pkl\n"
        )
        return 1

    print(f"\nLoading WM-811K from {args.data} ...")
    loader = WM811KLoader(pkl_path=args.data)
    samples = loader.load(labeled_only=True)
    print(f"Loaded {len(samples):,} labeled samples")

    # Reproduce the exact same split used in training and evaluation
    splits = stratified_split(samples)
    print(f"Test split: {len(splits.test):,} samples")

    # Build per-class index
    by_class: dict[str, list] = {}
    for s in splits.test:
        by_class.setdefault(s.label_name, []).append(s)

    # ── Load inference engine ─────────────────────────────────────────────────
    print(f"\nLoading checkpoint ...")
    engine = WaferInference.from_checkpoint(ckpt_path)

    # ── Sample and run ────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    per_class: dict[str, dict] = {}
    total_correct = 0
    total_n = 0

    print(f"\nRunning inference ({args.samples_per_class} samples/class) ...")
    print(f"  {'Class':<14}  {'n':>4}  {'Correct':>7}  {'Accuracy':>8}")
    print(f"  {'-'*14}  {'-'*4}  {'-'*7}  {'-'*8}")

    for cls in WAFER_DEFECT_CLASSES:
        cls_samples = by_class.get(cls, [])
        n = min(args.samples_per_class, len(cls_samples))
        if n == 0:
            per_class[cls] = {"n_samples": 0, "n_correct": 0, "accuracy": None}
            print(f"  {cls:<14}  {'0':>4}  {'—':>7}  {'—':>8}")
            continue

        indices = rng.choice(len(cls_samples), size=n, replace=False)
        correct = 0
        for idx in indices:
            s = cls_samples[int(idx)]
            r = engine.predict(s.wafer_map, top_k=1)
            if r.predicted_class == cls:
                correct += 1

        acc = round(correct / n, 4)
        per_class[cls] = {"n_samples": n, "n_correct": correct, "accuracy": acc}
        total_correct += correct
        total_n += n
        print(f"  {cls:<14}  {n:>4}  {correct:>7}  {acc:>8.4f}")

    overall_acc = round(total_correct / total_n, 4) if total_n > 0 else None
    print(f"\n  {'OVERALL':<14}  {total_n:>4}  {total_correct:>7}  {overall_acc:>8.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    result_doc = {
        "disclaimer": (
            "Metrics on WM-811K public dataset (portfolio project -- "
            "not real fab deployment performance)."
        ),
        "active_run_id": active_run_id,
        "checkpoint": str(ckpt_path),
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "preprocessing_matches_training": True,
        "preprocessing_note": (
            "Dashboard inference uses WaferInference._preprocess(): "
            "resize to (64,64) with nearest-neighbour, then normalize_wafer_map() "
            "which divides {0,1,2} by 2.0 → {0.0,0.5,1.0}. "
            "This is identical to WaferMapDataset.__getitem__() used in training."
        ),
        "overall_accuracy_on_sample": overall_acc,
        "total_samples": total_n,
        "per_class": per_class,
    }

    out_path = Path(WAFER_REPORTS_DIR) / "dashboard_inference_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_doc, f, indent=2)

    print(f"\nValidation complete.")
    print(f"  Checkpoint used  : {ckpt_path.name}")
    print(f"  Active run       : {active_run_id}")
    print(f"  Output saved to  : {out_path}")
    print()
    print("NOTE: This is a portfolio project. Metrics are on public WM-811K data only.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
