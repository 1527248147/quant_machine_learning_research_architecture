"""
Build daily_bars asset table from normalised OHLCV data.

daily_bars schema:
  date, sid (as columns or index),
  market.open, market.high, market.low, market.close,
  market.volume, market.amount, market.adj_factor
"""
from __future__ import annotations
import logging

import pandas as pd

from engine.schema.validators import validate_daily_bars

logger = logging.getLogger(__name__)

_MARKET_COL_MAP = {
    "open": "market.open",
    "high": "market.high",
    "low": "market.low",
    "close": "market.close",
    "volume": "market.volume",
    "amount": "market.amount",
    "adj_factor": "market.adj_factor",
}


def build_daily_bars(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the daily_bars asset table from a normalised OHLCV long-table.

    Parameters
    ----------
    ohlcv_df:
        Output of OHLCVSource.load() with columns
        [date, sid, open, high, low, close, volume, amount, adj_factor].

    Returns
    -------
    daily_bars DataFrame with columns
    [date, sid, market.open, …, market.adj_factor].
    """
    df = ohlcv_df

    # Rename to market.* namespace
    rename = {k: v for k, v in _MARKET_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Keep only the columns we need
    keep = ["date", "sid"] + [v for v in _MARKET_COL_MAP.values() if v in df.columns]
    df = df[keep]

    validate_daily_bars(df)

    logger.info("daily_bars: %d rows, %d unique sids", len(df), df["sid"].nunique())
    return df
