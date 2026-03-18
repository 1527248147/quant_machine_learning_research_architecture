"""
Base class for all models.

Every model must declare a contract that describes which labels it accepts.
The training preflight check uses this contract to validate the config before
any data is loaded or any training begins.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from engine.schema.contracts import ModelLabelContract  # noqa: F401 — re-export


class BaseModel(ABC):
    """Abstract base for all models."""

    name: str
    contract: ModelLabelContract

    @abstractmethod
    def build_model(self, input_dim: int, config: dict) -> None:
        """Instantiate the underlying model (e.g., nn.Module) from config."""

    @abstractmethod
    def fit(
        self,
        train_loader: Any,
        valid_loader: Any,
        config: dict,
        callbacks: Optional[Dict] = None,
    ) -> dict:
        """
        Train the model.

        Returns a dict of training info (e.g., best_epoch, best_val_loss).
        """

    @abstractmethod
    def predict(self, loader: Any) -> dict:
        """Return predictions as a dict of numpy arrays."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model checkpoint to *path*."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model checkpoint from *path*."""

    @abstractmethod
    def to_device(self, device: Any) -> None:
        """Move the model to the given device."""
