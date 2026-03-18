"""
Time-based data splitter.

Converts config date ranges into index-based masks / day-index ranges
that the trainer and dataset use to slice the memmap panel.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitBundle:
    """Index ranges (into the memmap date axis) for each split."""
    train_start_idx: int
    train_end_idx: int
    valid_start_idx: int
    valid_end_idx: int
    test_start_idx: int
    test_end_idx: int
    dates: List[str]

    @property
    def train_days(self) -> int:
        return self.train_end_idx - self.train_start_idx + 1

    @property
    def valid_days(self) -> int:
        return self.valid_end_idx - self.valid_start_idx + 1

    @property
    def test_days(self) -> int:
        return self.test_end_idx - self.test_start_idx + 1

    @property
    def train_plus_valid_start_idx(self) -> int:
        return self.train_start_idx

    @property
    def train_plus_valid_end_idx(self) -> int:
        return self.valid_end_idx


def _date_to_idx(dates: List[str], target: str, side: str = "left") -> int:
    """
    Find the index of *target* date in *dates* list.

    Parameters
    ----------
    dates : Sorted list of date strings (YYYY-MM-DD).
    target : Target date string.
    side : "left" for >= (start), "right" for < (end, exclusive).

    Returns
    -------
    Integer index into *dates*.
    """
    arr = np.array(dates)
    if side == "left":
        # First date >= target
        idx = int(np.searchsorted(arr, target, side="left"))
    else:
        # Last date < target  →  index = searchsorted(left) - 1
        idx = int(np.searchsorted(arr, target, side="left")) - 1
    return max(0, min(idx, len(dates) - 1))


def build_split_bundle(
    dates: List[str],
    config: dict,
    lookback: int = 60,
) -> SplitBundle:
    """
    Build a SplitBundle from config date ranges and the memmap date list.

    Config keys used (under ``training.split``):
        train_start, train_end, valid_start, valid_end, test_start, test_end

    All ranges are interpreted as [start, end) — left-closed, right-open.
    The returned indices are inclusive on both sides (for memmap slicing).

    The ``lookback`` parameter ensures the first usable day has enough history.
    """
    split_cfg = config["training"]["split"]

    train_s = _date_to_idx(dates, split_cfg["train_start"], "left")
    train_e = _date_to_idx(dates, split_cfg["train_end"], "right")
    valid_s = _date_to_idx(dates, split_cfg["valid_start"], "left")
    valid_e = _date_to_idx(dates, split_cfg["valid_end"], "right")
    test_s = _date_to_idx(dates, split_cfg["test_start"], "left")
    test_e = _date_to_idx(dates, split_cfg["test_end"], "right")

    # Ensure lookback constraint
    min_start = lookback - 1
    train_s = max(train_s, min_start)
    valid_s = max(valid_s, min_start)
    test_s = max(test_s, min_start)

    bundle = SplitBundle(
        train_start_idx=train_s,
        train_end_idx=train_e,
        valid_start_idx=valid_s,
        valid_end_idx=valid_e,
        test_start_idx=test_s,
        test_end_idx=test_e,
        dates=dates,
    )

    logger.info(
        "Split: train=[%d..%d] (%d days, %s~%s), "
        "valid=[%d..%d] (%d days, %s~%s), "
        "test=[%d..%d] (%d days, %s~%s)",
        bundle.train_start_idx, bundle.train_end_idx, bundle.train_days,
        dates[bundle.train_start_idx], dates[bundle.train_end_idx],
        bundle.valid_start_idx, bundle.valid_end_idx, bundle.valid_days,
        dates[bundle.valid_start_idx], dates[bundle.valid_end_idx],
        bundle.test_start_idx, bundle.test_end_idx, bundle.test_days,
        dates[bundle.test_start_idx], dates[bundle.test_end_idx],
    )

    return bundle
