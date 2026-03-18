"""
Base class for all view builders.

A view builder reads source data (panel_base + target_blocks) directly
via PyArrow column selection and produces the format a specific model needs
(e.g., 3D memmap tensor for LSTM, 2D matrix for LightGBM).

Each view builder handles its own data loading — it only reads the columns
it needs, so memory usage stays minimal regardless of panel_base size.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

from engine.io.paths import PathManager


class BaseViewBuilder(ABC):
    """Abstract base for all view builders."""

    name: str

    @abstractmethod
    def required_columns(self) -> List[str]:
        """Return column names from panel_base that this view needs."""

    @abstractmethod
    def build(
        self,
        paths: PathManager,
        config: dict,
    ) -> Path:
        """
        Build view artifacts (e.g., memmap files) from panel_base + target_blocks.

        Each view builder reads the source parquet files directly using
        PyArrow column selection, loading only the columns it needs.

        Returns
        -------
        Path to the output directory containing the built artifacts.
        """

    @abstractmethod
    def get_dataset(
        self,
        view_dir: Path,
        day_start: int,
        day_end: int,
        config: dict,
    ) -> Any:
        """
        Return a torch Dataset (or equivalent) for the given day index range.

        Parameters
        ----------
        view_dir : Directory containing the built artifacts (from ``build``).
        day_start, day_end : Inclusive day index range into the memmap.
        config : Full experiment config dict.
        """

    @abstractmethod
    def build_dataloader(self, dataset: Any, config: dict, shuffle: bool = True) -> Any:
        """Wrap a dataset into a DataLoader (or equivalent)."""
