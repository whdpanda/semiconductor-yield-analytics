"""Module B pipeline entry point: SPC + Process Anomaly Detection.

Placeholder — full implementation in Phase 2-3.
Run: python scripts/run_module_b_pipeline.py --config configs/module_b.yaml --stage full
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPC and anomaly detection pipeline")
    parser.add_argument("--config", default="configs/module_b.yaml")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "spc", "train_anomaly", "detect", "full"],
        default="full",
    )
    args = parser.parse_args()

    # TODO (Phase 2-3): implement full pipeline
    print(f"[run_module_b_pipeline] stage={args.stage}, config={args.config}")
    print("Not yet implemented — will be built in Phase 2-3.")


if __name__ == "__main__":
    main()
