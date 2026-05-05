"""Single-sample inference for WaferCNN (Module A).

Public API:
  InferenceResult   -- dataclass returned by WaferInference.predict()
  WaferInference    -- loads a WaferCNN checkpoint and runs predictions
  parse_wafer_input -- parse a raw numpy array / bytes to a wafer map ndarray
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from semiconductor_yield.config import MODELS_DIR, WAFER_DEFECT_CLASSES, WAFER_MAP_SIZE
from semiconductor_yield.models.wafer_cnn import WaferCNN
from semiconductor_yield.wafer.preprocess import normalize_wafer_map, resize_wafer_map


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Single-wafer prediction output."""

    predicted_class: str
    class_index: int
    confidence: float                   # softmax prob of the top-1 class, in [0, 1]
    top_k: list[tuple[str, float]]      # [(class_name, prob), ...] sorted descending
    preprocessed_map: np.ndarray        # (H, W) float32 in [0, 1] — what the model saw
    is_demo: bool = False               # True if model has random (untrained) weights


# ── Inference engine ───────────────────────────────────────────────────────────

class WaferInference:
    """Loads a WaferCNN and runs single-sample predictions.

    Use WaferInference.from_checkpoint() for a trained model, or
    WaferInference.demo() to get a randomly-initialised instance for UI
    demonstration (predictions are meaningless and labelled accordingly).

    Example::

        inf = WaferInference.from_checkpoint("outputs/models/wafer_cnn_best.pth")
        result = inf.predict(wafer_map_array)
        print(result.predicted_class, f"{result.confidence:.1%}")
    """

    def __init__(
        self,
        model: WaferCNN,
        class_names: list[str],
        device: torch.device,
        is_demo: bool = False,
    ) -> None:
        self._model = model
        self._class_names = class_names
        self._device = device
        self._is_demo = is_demo

    # ── Constructors ───────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str = MODELS_DIR / "wafer_cnn_best.pth",
        num_classes: int = len(WAFER_DEFECT_CLASSES),
        class_names: list[str] | None = None,
        device_str: str = "auto",
    ) -> "WaferInference":
        """Load a trained WaferCNN from a saved state-dict file.

        Args:
            checkpoint_path: Path to the .pth file produced by train_wafer_cnn.py.
            num_classes: Must match the saved model architecture (default 9).
            class_names: Ordered class name list. Defaults to WAFER_DEFECT_CLASSES.
            device_str: 'auto', 'cpu', or 'cuda'.

        Raises:
            FileNotFoundError: checkpoint_path does not exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Train the model first:\n"
                "  python scripts/train_wafer_cnn.py"
            )

        device = _resolve_device(device_str)
        model = WaferCNN(num_classes=num_classes, dropout=0.0)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()

        if class_names is None:
            class_names = list(WAFER_DEFECT_CLASSES)
        return cls(model=model, class_names=class_names, device=device, is_demo=False)

    @classmethod
    def demo(
        cls,
        num_classes: int = len(WAFER_DEFECT_CLASSES),
        class_names: list[str] | None = None,
    ) -> "WaferInference":
        """Create a randomly-initialised WaferCNN for UI demo purposes.

        Predictions from a demo instance are MEANINGLESS.  Always surface
        result.is_demo == True to the user so they know.
        """
        device = torch.device("cpu")
        model = WaferCNN(num_classes=num_classes, dropout=0.0)
        model.eval()
        if class_names is None:
            class_names = list(WAFER_DEFECT_CLASSES)
        return cls(model=model, class_names=class_names, device=device, is_demo=True)

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict(self, wafer_map: np.ndarray, top_k: int = 5) -> InferenceResult:
        """Run inference on a single wafer map.

        Args:
            wafer_map: 2-D float or int array with values in {0, 1, 2}.
                       Arbitrary spatial resolution — resized to (64, 64) internally
                       using nearest-neighbour interpolation.
            top_k: Number of classes to include in the ranked probability output.

        Returns:
            InferenceResult with predicted class, confidence score, and top-k probs.
        """
        top_k = min(max(1, top_k), len(self._class_names))
        preprocessed = _preprocess(wafer_map)
        tensor = (
            torch.from_numpy(preprocessed)
            .unsqueeze(0)   # (H, W) → (1, H, W)
            .unsqueeze(0)   # (1, H, W) → (1, 1, H, W)
            .to(self._device)
        )

        with torch.no_grad():
            logits = self._model(tensor)                          # (1, C)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()    # (C,)

        probs_np = probs.numpy()
        ranked = probs_np.argsort()[::-1]

        top_k_list: list[tuple[str, float]] = [
            (self._class_names[i], float(probs_np[i]))
            for i in ranked[:top_k]
        ]
        best_idx = int(ranked[0])

        return InferenceResult(
            predicted_class=self._class_names[best_idx],
            class_index=best_idx,
            confidence=float(probs_np[best_idx]),
            top_k=top_k_list,
            preprocessed_map=preprocessed,
            is_demo=self._is_demo,
        )

    def predict_batch(
        self,
        wafer_maps: list[np.ndarray],
        top_k: int = 5,
    ) -> list[InferenceResult]:
        """Run predict() over a list of wafer maps."""
        return [self.predict(wm, top_k=top_k) for wm in wafer_maps]


# ── Input parsing ──────────────────────────────────────────────────────────────

def parse_wafer_input(data: np.ndarray | bytes, filename: str = "") -> np.ndarray:
    """Parse a raw user-supplied input into a (H, W) wafer map array.

    Supported formats:
      - numpy array    — returned as-is (must be 2-D)
      - .npy bytes     — numpy.load from in-memory buffer
      - .pkl bytes     — pickle.loads; result must be ndarray or dict{'wafer_map': …}
      - .csv bytes     — numpy.loadtxt, comma-separated grid

    Args:
        data: Raw input — either an ndarray or bytes from a file upload.
        filename: Original filename (used to detect format from extension).

    Returns:
        float32 2-D array with values in {0.0, 1.0, 2.0}.

    Raises:
        ValueError: Unrecognised format or wrong array dimensionality.
    """
    if isinstance(data, np.ndarray):
        return _coerce_wafer_map(data)

    ext = Path(filename).suffix.lower()

    if ext == ".npy":
        arr = np.load(io.BytesIO(data))
        return _coerce_wafer_map(arr)

    if ext in (".pkl", ".pickle"):
        obj = pickle.loads(data)  # noqa: S301  (user-supplied data in UI context)
        if isinstance(obj, np.ndarray):
            return _coerce_wafer_map(obj)
        if isinstance(obj, dict) and "wafer_map" in obj:
            return _coerce_wafer_map(np.array(obj["wafer_map"]))
        raise ValueError(
            "Pickle file must contain an ndarray or a dict with key 'wafer_map'."
        )

    if ext == ".csv":
        arr = np.loadtxt(io.BytesIO(data), delimiter=",")
        return _coerce_wafer_map(arr)

    raise ValueError(
        f"Unsupported file extension '{ext}'. Supported: .npy, .pkl, .csv"
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _preprocess(wafer_map: np.ndarray) -> np.ndarray:
    """Resize to WAFER_MAP_SIZE and normalise {0,1,2} → {0.0,0.5,1.0}."""
    arr = np.asarray(wafer_map, dtype=np.float32)
    if arr.shape[:2] != WAFER_MAP_SIZE:
        arr = resize_wafer_map(arr, WAFER_MAP_SIZE)
    return normalize_wafer_map(arr)


def _coerce_wafer_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"Wafer map must be a 2-D array, got shape {arr.shape}"
        )
    return arr


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
