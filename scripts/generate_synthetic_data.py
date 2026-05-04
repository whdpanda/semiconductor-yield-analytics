"""Generate synthetic SPC process data for Module B.

Placeholder — full implementation in Phase 2.
Run: python scripts/generate_synthetic_data.py
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SPC data")
    parser.add_argument("--config", default="configs/module_b.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output", default="data/synthetic/spc_with_anomalies.csv")
    args = parser.parse_args()

    # TODO (Phase 2): implement data generation
    print(f"[generate_synthetic_data] config={args.config}, seed={args.seed}")
    print("Not yet implemented — will be built in Phase 2.")


if __name__ == "__main__":
    main()
