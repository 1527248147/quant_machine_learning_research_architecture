"""
Build status_intervals: daily suspension flags derived from OHLCV volume.

status_intervals schema:
  date, sid, is_suspended (bool)

A (date, sid) row is marked as suspended when volume == 0 in the OHLCV data.
Only suspended rows are stored (the table is sparse — normal days are absent).
The panel builder fills absent rows as is_suspended=False.
"""
from __future__ import annotations
import logging

import pandas as pd

from engine.core.constants import SUSPENSION_VOLUME_THRESHOLD

logger = logging.getLogger(__name__)


def build_status_intervals(daily_bars: pd.DataFrame) -> pd.DataFrame:
    """
    Derive suspension flags from daily_bars (volume == 0 ↔ suspended).

    Parameters
    ----------
    daily_bars:
        Asset table with columns [date, sid, market.volume, …].

    Returns
    -------
    Sparse DataFrame with columns [date, sid, is_suspended].
    Only rows where is_suspended=True are included.
    """
    if "market.volume" not in daily_bars.columns:
        logger.warning(
            "status_intervals: 'market.volume' not found; returning empty table."
        )
        return pd.DataFrame(columns=["date", "sid", "is_suspended"])

    suspended = daily_bars[
        daily_bars["market.volume"] <= SUSPENSION_VOLUME_THRESHOLD
    ][["date", "sid"]].copy()
    suspended["is_suspended"] = True

    logger.info(
        "status_intervals: %d suspended (date, sid) pairs out of %d total",
        len(suspended),
        len(daily_bars),
    )
    return suspended.reset_index(drop=True)
