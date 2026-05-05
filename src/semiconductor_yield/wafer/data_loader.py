"""WM-811K wafer map dataset loader.

Handles loading and initial parsing of LSWMD.pkl. Does NOT perform training-time
augmentation or train/val/test splitting — those belong in the training pipeline.

The WM-811K pkl file is NOT bundled with this project.
Download instructions: see docs/data_contract.md or README.md.
"""

from __future__ import annotations

import importlib
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from semiconductor_yield.config import WAFER_DEFECT_CLASSES, WAFER_MAP_SIZE, WM811K_PKL

# ── Label mapping ──────────────────────────────────────────────────────────────

LABEL_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(WAFER_DEFECT_CLASSES)}
IDX_TO_LABEL: dict[int, str] = {i: name for name, i in LABEL_TO_IDX.items()}


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WaferSample:
    """A single parsed and (optionally resized) wafer map sample."""

    wafer_map: np.ndarray   # shape (H, W), float32, values in {0.0, 1.0, 2.0}
    label: int              # integer class index
    label_name: str         # human-readable class name
    lot_name: str
    wafer_index: int
    split: str              # "train" or "test" per the original dataset split field


# ── Legacy pickle compatibility ────────────────────────────────────────────────

def _install_legacy_pandas_pickle_aliases() -> None:
    """Register old pandas.indexes.* module paths that no longer exist.

    LSWMD.pkl was serialised with an old pandas version that stored objects
    under 'pandas.indexes.*'. Modern pandas moved everything to
    'pandas.core.indexes.*'. Registering aliases in sys.modules lets the
    unpickler resolve the old paths without requiring a pandas downgrade.
    """
    import pandas.core.indexes as _core_indexes

    sys.modules.setdefault("pandas.indexes", _core_indexes)

    for _name in (
        "base", "range", "multi", "datetimes",
        "timedeltas", "period", "category", "interval",
    ):
        try:
            _mod = importlib.import_module(f"pandas.core.indexes.{_name}")
            sys.modules.setdefault(f"pandas.indexes.{_name}", _mod)
        except ModuleNotFoundError:
            pass


def _load_legacy_pickle(path: Path) -> pd.DataFrame:
    """Load a potentially legacy-pandas pickle, with automatic compatibility fallback.

    Strategy:
        1. Try pd.read_pickle(path)  — works for most modern pickles.
        2. If that raises ModuleNotFoundError (old 'pandas.indexes' paths) or
           UnicodeDecodeError (Python-2-era pickle with non-ASCII bytes), install
           legacy module aliases and retry with pickle.load(..., encoding='latin1').
           Both errors indicate LSWMD.pkl was serialised with an old pandas/Python 2.
    """
    try:
        return pd.read_pickle(path)
    except (ModuleNotFoundError, UnicodeDecodeError) as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name != "pandas.indexes":
            raise

        logger.warning(
            f"pd.read_pickle failed for legacy WM-811K pickle ({type(exc).__name__}); "
            "retrying with latin1 compatibility fallback."
        )
        _install_legacy_pandas_pickle_aliases()

        with open(path, "rb") as fh:
            return pickle.load(fh, encoding="latin1")


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_failure_type(raw: object) -> Optional[str]:
    """Extract a failure-type string from the nested-list format in LSWMD.pkl.

    The WM-811K dataset encodes failureType as e.g. [['Center']], [['none']],
    [['']] (unlabeled), or occasionally as a plain string in pre-processed forks.

    Returns:
        Class name string, or None if the entry is unlabeled / empty.
    """
    if raw is None:
        return None
    # Plain string (some pre-processed forks of the dataset)
    if isinstance(raw, str):
        s = raw.strip()
        return s if s else None
    # Expected nested-list format: [['Center']] or [['']]
    if isinstance(raw, (list, np.ndarray)):
        try:
            inner = raw[0]
            value = inner[0] if isinstance(inner, (list, np.ndarray)) else inner
            s = str(value).strip()
            return s if s else None
        except (IndexError, TypeError):
            return None
    return None


def _resize_wafer_map(wmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a wafer map to (H, W) using nearest-neighbour interpolation.

    Nearest-neighbour is mandatory here: the three discrete values (0, 1, 2)
    must not be blended by bilinear or bicubic resampling.
    """
    from PIL import Image  # deferred import — pillow is a project dependency

    img = Image.fromarray(wmap.astype(np.uint8), mode="L")
    img = img.resize((size[1], size[0]), resample=Image.Resampling.NEAREST)
    return np.array(img, dtype=np.float32)


# ── Main loader class ──────────────────────────────────────────────────────────

class WM811KLoader:
    """Loads and parses the WM-811K wafer map dataset from LSWMD.pkl.

    The loader is intentionally thin: it parses labels, optionally resizes maps,
    and returns clean Python objects. Further splitting and augmentation happen
    in the training pipeline.

    Example:
        loader = WM811KLoader()
        samples = loader.load()          # list[WaferSample], labeled only
        df_raw  = loader.load_raw()      # raw DataFrame for EDA
        dist    = loader.class_distribution()
    """

    def __init__(
        self,
        pkl_path: Path = WM811K_PKL,
        target_size: tuple[int, int] = WAFER_MAP_SIZE,
        resize: bool = True,
    ) -> None:
        self.pkl_path = Path(pkl_path)
        self.target_size = target_size
        self.resize = resize

    # ── Public methods ─────────────────────────────────────────────────────────

    def load_raw(self) -> pd.DataFrame:
        """Return the raw DataFrame from LSWMD.pkl without any filtering or resizing."""
        self._assert_file_exists()
        logger.info(f"Loading WM-811K from {self.pkl_path} ...")
        df = _load_legacy_pickle(self.pkl_path)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame from {self.pkl_path}, "
                f"got {type(df).__name__}. "
                "Ensure you downloaded the correct LSWMD.pkl from Kaggle."
            )
        logger.info(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")
        return df

    def load(self, labeled_only: bool = True) -> list[WaferSample]:
        """Load, parse, and optionally resize wafer map samples.

        Args:
            labeled_only: When True (default), discard wafers where failureType is
                empty (unlabeled). The dataset contains ~811k total wafers but only
                ~172k carry a defect-pattern label.

        Returns:
            List of WaferSample with parsed labels and, if resize=True, maps
            uniformly resized to self.target_size.
        """
        df = self.load_raw()
        samples: list[WaferSample] = []
        n_unlabeled = 0
        n_unknown = 0

        for _, row in df.iterrows():
            label_name = _parse_failure_type(row.get("failureType"))

            if label_name is None or label_name == "":
                n_unlabeled += 1
                if labeled_only:
                    continue
                label_name = "none"

            if label_name not in LABEL_TO_IDX:
                n_unknown += 1
                logger.debug(f"Unknown class '{label_name}' — skipped")
                continue

            wmap = np.array(row["waferMap"], dtype=np.float32)
            if self.resize and wmap.shape[:2] != self.target_size:
                wmap = _resize_wafer_map(wmap, self.target_size)

            split = _parse_split_label(row.get("trianTestLabel"))

            samples.append(
                WaferSample(
                    wafer_map=wmap,
                    label=LABEL_TO_IDX[label_name],
                    label_name=label_name,
                    lot_name=str(row.get("lotName", "")),
                    wafer_index=int(row.get("waferIndex", -1)),
                    split=split,
                )
            )

        logger.info(
            f"Parsed {len(samples):,} samples "
            f"(skipped {n_unlabeled:,} unlabeled, {n_unknown} unknown-class)"
        )
        return samples

    def class_distribution(self) -> pd.Series:
        """Return class counts from the labeled subset (fast — no resizing)."""
        df = self.load_raw()
        labels = df["failureType"].apply(_parse_failure_type)
        labels = labels.dropna()
        labels = labels[labels != ""]
        return labels.value_counts()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _assert_file_exists(self) -> None:
        if not self.pkl_path.exists():
            raise FileNotFoundError(
                f"\nWM-811K dataset not found at: {self.pkl_path}\n\n"
                "To obtain the data:\n"
                "  1. Create a free Kaggle account at https://www.kaggle.com\n"
                "  2. Go to https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map\n"
                "  3. Download LSWMD.pkl  (~350 MB)\n"
                "  4. Place it at: data/raw/wm811k/LSWMD.pkl\n\n"
                "The file is NOT included in this repository due to its size."
            )


def _parse_split_label(raw: object) -> str:
    """Return 'train' or 'test' from the trianTestLabel field."""
    if raw is None:
        return "train"
    s = str(raw)
    if "Test" in s or "test" in s:
        return "test"
    return "train"
