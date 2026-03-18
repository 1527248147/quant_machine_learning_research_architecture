"""
Base class for all target recipes.

A recipe takes panel_base as input and produces a target_block with exactly
three columns per target:
  - label.<target_name>        : the label value
  - label_valid.<target_name>  : bool, whether this label is usable
  - label_reason.<target_name> : str, reason if invalid (empty string if valid)

Recipes must NOT modify panel_base.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd

from engine.core.exceptions import TargetBuildError


class BaseTargetRecipe(ABC):
    """Abstract base for all target recipes."""

    name: str

    @abstractmethod
    def required_columns(self) -> List[str]:
        """Return list of panel_base columns this recipe needs."""

    @abstractmethod
    def compute(self, panel_base: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the target_block.

        Parameters
        ----------
        panel_base : DataFrame with index=[date, sid].

        Returns
        -------
        DataFrame with index=[date, sid] and columns:
            label.<name>, label_valid.<name>, label_reason.<name>
        """

    def class_names(self) -> Dict[int, str]:
        """Return {class_int: display_name} for classification recipes. Empty for regression."""
        return {}

    def validate_inputs(self, panel_base: pd.DataFrame) -> None:
        """Check that panel_base has all required columns. Raises TargetBuildError."""
        missing = [c for c in self.required_columns() if c not in panel_base.columns]
        if missing:
            raise TargetBuildError(
                f"[{self.name}] panel_base is missing columns required by this recipe: {missing}"
            )
