from __future__ import annotations

from abc import ABC, abstractmethod

import joblib
import numpy as np


class TemporalAblationModel(ABC):
    """Base interface for temporal inundation ablation models."""

    model_type: str

    def __init__(self, seed: int):
        self.seed = seed
        self.model = None
        self.residuals_: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_samples(self, X: np.ndarray) -> np.ndarray:
        """Return predictive samples with shape (n_samples, n_obs)."""
        point_pred = self.predict(X)
        residuals = self._residual_distribution()
        if residuals.size < 2:
            return np.expand_dims(point_pred, axis=0)
        rng = np.random.default_rng(self.seed)
        draws = rng.choice(residuals, size=(100, len(point_pred)), replace=True)
        return point_pred[np.newaxis, :] + draws

    def _store_training_residuals(self, X: np.ndarray, y: np.ndarray) -> None:
        residuals = np.asarray(y, dtype=np.float64) - np.asarray(self.predict(X), dtype=np.float64)
        residuals = residuals[np.isfinite(residuals)]
        self.residuals_ = residuals
        if self.model is not None:
            setattr(self.model, "_inflow_training_residuals", residuals)

    def _residual_distribution(self) -> np.ndarray:
        if self.residuals_ is not None:
            return np.asarray(self.residuals_, dtype=np.float64)
        if self.model is not None and hasattr(self.model, "_inflow_training_residuals"):
            return np.asarray(self.model._inflow_training_residuals, dtype=np.float64)
        return np.asarray([], dtype=np.float64)

    def save_weights(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        joblib.dump(self.model, path)


AblationModel = TemporalAblationModel
