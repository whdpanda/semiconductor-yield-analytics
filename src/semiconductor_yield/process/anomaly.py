"""ML-based anomaly detectors for semiconductor process data (Module B).

Two detectors are provided:

* **IsolationForestDetector** — sklearn Isolation Forest; multivariate,
  unsupervised, fast inference. Anomaly score = negated ``decision_function``
  (higher ↔ more anomalous).

* **AutoencoderDetector** — PyTorch MLP autoencoder; anomaly score = per-sample
  mean squared reconstruction error. Threshold set at the 95th percentile of
  training-set reconstruction errors.

Both detectors:
  - Are trained *unsupervised* (no labels used during ``fit``).
  - Internally standardise features with ``StandardScaler``.
  - Serialize to / deserialize from a single ``.joblib`` file.

All evaluation against ground-truth labels is done *outside* this module
(see ``scripts/evaluate_process_anomaly.py``) to keep training and
evaluation cleanly separated.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ── Shared result container ────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    """Scored output from a fitted anomaly detector."""

    scores: np.ndarray       # anomaly score per sample (higher = more anomalous)
    labels_pred: np.ndarray  # binary 0/1: 1 = predicted anomaly
    threshold: float         # score threshold used for binary prediction
    feature_cols: list[str]  # features used (in order)


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseDetector(ABC):
    """Common interface for all anomaly detectors."""

    @abstractmethod
    def fit(self, X: np.ndarray, feature_cols: list[str] | None = None) -> "BaseDetector":
        """Train the detector unsupervised on clean/mixed process data."""

    @abstractmethod
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return a 1-D score array; higher = more anomalous."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels (1=anomaly) using the fitted threshold."""
        return (self.anomaly_scores(X) >= self._threshold).astype(int)

    def score_and_predict(self, X: np.ndarray) -> AnomalyResult:
        scores = self.anomaly_scores(X)
        labels = (scores >= self._threshold).astype(int)
        return AnomalyResult(
            scores=scores,
            labels_pred=labels,
            threshold=self._threshold,
            feature_cols=list(getattr(self, "_feature_cols", [])),
        )

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize the trained detector to a single .joblib file."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseDetector":
        """Deserialize a trained detector from a .joblib file."""


# ── Isolation Forest ───────────────────────────────────────────────────────────

class IsolationForestDetector(BaseDetector):
    """Multivariate anomaly detector based on sklearn's Isolation Forest.

    Anomaly score = − decision_function value (negated so that higher means
    more anomalous, consistent with AutoencoderDetector).

    Args:
        contamination: Expected fraction of anomalies in training data.
                       Used only to set the internal threshold; training is
                       still unsupervised.
        n_estimators: Number of base estimators (trees).
        random_state: Reproducibility seed.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._threshold: float = 0.0
        self._feature_cols: list[str] = []

    def fit(self, X: np.ndarray, feature_cols: list[str] | None = None) -> "IsolationForestDetector":
        self._feature_cols = list(feature_cols or [])
        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X)

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self._model.fit(X_s)

        # Set threshold at the 95th percentile of training anomaly scores
        scores = self.anomaly_scores(X)
        self._threshold = float(np.percentile(scores, 95))

        logger.info(
            f"[IF] fit on {X.shape[0]} samples × {X.shape[1]} features | "
            f"threshold={self._threshold:.4f}"
        )
        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X)
        # decision_function: lower (more negative) = more anomalous
        # negate so higher = more anomalous
        return -self._model.decision_function(X_s)

    def feature_importances(self) -> np.ndarray:
        """Mean split-based feature importance across all trees."""
        return np.mean(
            [e.feature_importances_ for e in self._model.estimators_], axis=0
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        payload = {
            "kind": "isolation_forest",
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "scaler": self._scaler,
            "model": self._model,
            "threshold": self._threshold,
            "feature_cols": self._feature_cols,
        }
        joblib.dump(payload, path)
        logger.info(f"[IF] saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "IsolationForestDetector":
        payload = joblib.load(Path(path))
        obj = cls(
            contamination=payload["contamination"],
            n_estimators=payload["n_estimators"],
            random_state=payload["random_state"],
        )
        obj._scaler = payload["scaler"]
        obj._model = payload["model"]
        obj._threshold = payload["threshold"]
        obj._feature_cols = payload.get("feature_cols", [])
        return obj


# ── PyTorch MLP Autoencoder ────────────────────────────────────────────────────

def _make_mlp(dims: list[int]) -> "torch.nn.Sequential":
    """Build an MLP with ReLU activations between layers; no activation on output."""
    import torch.nn as nn

    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class _MLPAutoencoder:
    """Thin wrapper that holds encoder + decoder as a single nn.Module."""

    def __new__(cls, input_dim: int, hidden_dims: tuple[int, ...]) -> "torch.nn.Module":
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                enc_dims = [input_dim] + list(hidden_dims)
                dec_dims = list(reversed(hidden_dims)) + [input_dim]
                self.encoder = _make_mlp(enc_dims)
                self.decoder = _make_mlp(dec_dims)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                return self.decoder(self.encoder(x))

        return Net()


class AutoencoderDetector(BaseDetector):
    """Anomaly detector based on a PyTorch MLP autoencoder.

    Anomaly score = mean squared reconstruction error per sample.
    Threshold = 95th percentile of training-set reconstruction errors.

    Architecture: input → *hidden_dims (encoder) → reversed(*hidden_dims) → input (decoder).

    Args:
        hidden_dims: Encoder layer sizes. Decoder mirrors them.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (32, 16, 8),
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._threshold: float = 0.0
        self._feature_cols: list[str] = []

    def fit(self, X: np.ndarray, feature_cols: list[str] | None = None) -> "AutoencoderDetector":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._feature_cols = list(feature_cols or [])
        self._input_dim = X.shape[1]

        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X).astype(np.float32)

        self._net = _MLPAutoencoder(self._input_dim, self.hidden_dims)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_s)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                loss = criterion(self._net(batch), batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug(f"[AE] epoch {epoch+1}/{self.epochs}  loss={epoch_loss/len(loader):.5f}")

        scores = self.anomaly_scores(X)
        self._threshold = float(np.percentile(scores, 95))

        logger.info(
            f"[AE] fit on {X.shape[0]} samples × {X.shape[1]} features | "
            f"epochs={self.epochs} | threshold={self._threshold:.6f}"
        )
        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        import torch

        X_s = self._scaler.transform(X).astype(np.float32)
        self._net.eval()
        with torch.no_grad():
            recon = self._net(torch.from_numpy(X_s)).numpy()
        return np.mean((X_s - recon) ** 2, axis=1)

    def per_feature_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error per feature column (shape: p)."""
        import torch

        X_s = self._scaler.transform(X).astype(np.float32)
        self._net.eval()
        with torch.no_grad():
            recon = self._net(torch.from_numpy(X_s)).numpy()
        return np.mean((X_s - recon) ** 2, axis=0)

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        buf = io.BytesIO()
        torch.save(self._net.state_dict(), buf)
        payload = {
            "kind": "autoencoder",
            "input_dim": self._input_dim,
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "scaler": self._scaler,
            "threshold": self._threshold,
            "feature_cols": self._feature_cols,
            "net_state_bytes": buf.getvalue(),
        }
        joblib.dump(payload, path)
        logger.info(f"[AE] saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "AutoencoderDetector":
        import torch

        payload = joblib.load(Path(path))
        obj = cls(
            hidden_dims=payload["hidden_dims"],
            epochs=payload["epochs"],
            batch_size=payload["batch_size"],
            lr=payload["lr"],
        )
        obj._input_dim = payload["input_dim"]
        obj._scaler = payload["scaler"]
        obj._threshold = payload["threshold"]
        obj._feature_cols = payload.get("feature_cols", [])
        obj._net = _MLPAutoencoder(payload["input_dim"], payload["hidden_dims"])
        state_buf = io.BytesIO(payload["net_state_bytes"])
        obj._net.load_state_dict(torch.load(state_buf, weights_only=True))
        return obj
