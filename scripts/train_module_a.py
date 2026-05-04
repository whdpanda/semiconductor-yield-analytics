"""Module A training entry point: Wafer Map Defect Classification.

Placeholder — full implementation in Phase 5.
Run: python scripts/train_module_a.py --config configs/module_a.yaml
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train wafer map defect classifier")
    parser.add_argument("--config", default="configs/module_a.yaml")
    parser.add_argument("--stage", choices=["preprocess", "train", "evaluate", "all"], default="all")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # TODO (Phase 4-5): implement preprocessing and training
    print(f"[train_module_a] stage={args.stage}, config={args.config}")
    print("Not yet implemented — will be built in Phase 4-5.")


if __name__ == "__main__":
    main()
