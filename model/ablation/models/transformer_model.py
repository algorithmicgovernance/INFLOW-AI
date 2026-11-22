from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from processing.config import get_cfg

from .base_model import AblationModel


@dataclass
class TransformerInput:
    """Encoder history and decoder meteorological forecast tokens."""

    history: np.ndarray  # (sample, historical period, observed feature)
    future: np.ndarray  # (sample, ECMWF lead window, forecast feature)

    def __len__(self) -> int:
        return int(self.history.shape[0])


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError(
            "The transformer ablation model requires PyTorch. Install project requirements "
            "or run `python -m pip install torch`."
        ) from exc
    return torch


def _build_network(history_dim: int, future_dim: int, horizon_count: int, settings: dict[str, Any]):
    torch = _torch()
    nn = torch.nn
    d_model = int(settings["d_model"])
    nhead = int(settings["nhead"])
    if d_model % nhead:
        raise ValueError("Transformer d_model must be divisible by nhead.")

    class ForecastTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.history_projection = nn.Linear(history_dim, d_model)
            self.future_projection = nn.Linear(future_dim, d_model)
            self.history_position = nn.Parameter(
                torch.zeros(1, int(settings["history_length"]), d_model)
            )
            self.future_position = nn.Parameter(torch.zeros(1, 4, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=int(settings["feedforward_dim"]),
                dropout=float(settings["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=int(settings["feedforward_dim"]),
                dropout=float(settings["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, int(settings["encoder_layers"]))
            self.decoder = nn.TransformerDecoder(decoder_layer, int(settings["decoder_layers"]))
            self.head = nn.Sequential(
                nn.LayerNorm(4 * d_model),
                nn.Linear(4 * d_model, int(settings["feedforward_dim"])),
                nn.GELU(),
                nn.Dropout(float(settings["dropout"])),
                nn.Linear(int(settings["feedforward_dim"]), horizon_count),
            )

        def forward(self, history, future):
            history_tokens = self.history_projection(history) + self.history_position[:, : history.shape[1]]
            future_tokens = self.future_projection(future) + self.future_position[:, : future.shape[1]]
            memory = self.encoder(history_tokens)
            decoded = self.decoder(future_tokens, memory)
            return self.head(decoded.flatten(start_dim=1))

    return ForecastTransformer()


class TransformerAblationModel(AblationModel):
    """Non-autoregressive encoder-decoder that predicts every horizon jointly."""

    model_type = "transformer"
    is_multi_horizon = True

    def __init__(self, seed: int):
        super().__init__(seed=seed)
        prefix = "ablation.pipeline.transformer"
        self.settings = {
            "history_length": int(get_cfg(f"{prefix}.history_length", 12)),
            "d_model": int(get_cfg(f"{prefix}.d_model", 64)),
            "nhead": int(get_cfg(f"{prefix}.nhead", 4)),
            "encoder_layers": int(get_cfg(f"{prefix}.encoder_layers", 2)),
            "decoder_layers": int(get_cfg(f"{prefix}.decoder_layers", 2)),
            "feedforward_dim": int(get_cfg(f"{prefix}.feedforward_dim", 128)),
            "dropout": float(get_cfg(f"{prefix}.dropout", 0.1)),
            "learning_rate": float(get_cfg(f"{prefix}.learning_rate", 1e-3)),
            "weight_decay": float(get_cfg(f"{prefix}.weight_decay", 1e-4)),
            "batch_size": int(get_cfg(f"{prefix}.batch_size", 32)),
            "epochs": int(get_cfg(f"{prefix}.epochs", 200)),
            "patience": int(get_cfg(f"{prefix}.patience", 20)),
            "validation_fraction": float(get_cfg(f"{prefix}.validation_fraction", 0.15)),
            "predictive_samples": int(get_cfg(f"{prefix}.predictive_samples", 30)),
        }
        self.history_dim = 0
        self.future_dim = 0
        self.horizon_count = 0

    @staticmethod
    def _validate(X: TransformerInput, y: np.ndarray | None = None) -> None:
        if not isinstance(X, TransformerInput):
            raise TypeError("TransformerAblationModel expects TransformerInput.")
        if X.history.ndim != 3 or X.future.ndim != 3:
            raise ValueError("Transformer history and future inputs must both be three-dimensional.")
        if len(X) != X.future.shape[0] or (y is not None and len(X) != len(y)):
            raise ValueError("Transformer input and target sample counts differ.")
        if X.future.shape[1] != 4:
            raise ValueError("Transformer decoder requires the four ECMWF 10-day forecast windows.")

    def fit(self, X: TransformerInput, y: np.ndarray) -> None:
        self._validate(X, y)
        if y.ndim != 2:
            raise ValueError("Transformer targets must have shape (samples, forecast horizons).")
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.history_dim = int(X.history.shape[2])
        self.future_dim = int(X.future.shape[2])
        self.horizon_count = int(y.shape[1])
        self.model = _build_network(self.history_dim, self.future_dim, self.horizon_count, self.settings)

        history = torch.as_tensor(X.history, dtype=torch.float32)
        future = torch.as_tensor(X.future, dtype=torch.float32)
        targets = torch.as_tensor(y, dtype=torch.float32)
        validation_count = max(1, int(round(len(y) * self.settings["validation_fraction"])))
        if validation_count >= len(y):
            raise ValueError("Not enough training rows for a temporal validation split.")
        split = len(y) - validation_count
        train_dataset = torch.utils.data.TensorDataset(history[:split], future[:split], targets[:split])
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(int(self.settings["batch_size"]), split),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.settings["learning_rate"]),
            weight_decay=float(self.settings["weight_decay"]),
        )
        loss_fn = torch.nn.MSELoss()
        best_state = None
        best_loss = float("inf")
        stale_epochs = 0
        for _ in range(int(self.settings["epochs"])):
            self.model.train()
            for batch_history, batch_future, batch_targets in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(batch_history, batch_future), batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_fn(self.model(history[split:], future[split:]), targets[split:]).item()
                )
            if validation_loss < best_loss - 1e-7:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(self.settings["patience"]):
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X: TransformerInput) -> np.ndarray:
        self._validate(X)
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        torch = _torch()
        self.model.eval()
        with torch.no_grad():
            result = self.model(
                torch.as_tensor(X.history, dtype=torch.float32),
                torch.as_tensor(X.future, dtype=torch.float32),
            )
        return result.cpu().numpy()

    def predict_samples(self, X: TransformerInput) -> np.ndarray:
        """Monte-Carlo dropout samples with shape (draw, observation, horizon)."""
        self._validate(X)
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        torch = _torch()
        history = torch.as_tensor(X.history, dtype=torch.float32)
        future = torch.as_tensor(X.future, dtype=torch.float32)
        self.model.train()
        draws = []
        with torch.no_grad():
            for _ in range(max(1, int(self.settings["predictive_samples"]))):
                draws.append(self.model(history, future).cpu().numpy())
        self.model.eval()
        return np.stack(draws, axis=0)

    def save_weights(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        torch = _torch()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "settings": self.settings,
                "history_dim": self.history_dim,
                "future_dim": self.future_dim,
                "horizon_count": self.horizon_count,
                "seed": self.seed,
            },
            path,
        )

    def load_weights(self, path: str) -> None:
        torch = _torch()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.settings = dict(payload["settings"])
        self.history_dim = int(payload["history_dim"])
        self.future_dim = int(payload["future_dim"])
        self.horizon_count = int(payload["horizon_count"])
        self.model = _build_network(self.history_dim, self.future_dim, self.horizon_count, self.settings)
        self.model.load_state_dict(payload["state_dict"])

