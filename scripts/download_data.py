"""Data download helper.

WM-811K: requires a Kaggle account. Download LSWMD.pkl manually from
  https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
  and place it at: data/raw/wm811k/LSWMD.pkl

UCI SECOM: downloads automatically via urllib.
  https://archive.ics.uci.edu/dataset/179/secom
"""

import argparse
import urllib.request
from pathlib import Path

SECOM_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
SECOM_FILES = ["secom.data", "secom_labels.data"]


def download_secom(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for fname in SECOM_FILES:
        url = f"{SECOM_BASE}/{fname}"
        out = dest / fname
        if out.exists():
            print(f"  Already exists: {out}")
            continue
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, out)
        print(f"  Saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download project datasets")
    parser.add_argument("--dataset", choices=["secom", "all"], default="secom")
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    if args.dataset in ("secom", "all"):
        print("Downloading UCI SECOM ...")
        download_secom(root / "data" / "raw" / "secom")

    if args.dataset == "all":
        print("\nNote: WM-811K must be downloaded manually from Kaggle.")
        print("  URL: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map")
        print("  Place LSWMD.pkl at: data/raw/wm811k/LSWMD.pkl")


if __name__ == "__main__":
    main()
