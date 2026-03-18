"""
Build trading_calendar from OHLCV date coverage.

The standard input to the framework is only the OHLCV parquet directory and the
factor parquet directory. The trading calendar is always derived from the unique
dates present in the OHLCV data via ``build_trading_calendar_from_ohlcv``.

``build_trading_calendar_from_csv`` is kept as a utility for ad-hoc use only
and must NOT be called from the main pipeline.

trading_calendar schema:
  date    : Timestamp (trading day)
  is_open : bool (True for all rows — only open/trading days are stored)
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.schema.validators import validate_trading_calendar

logger = logging.getLogger(__name__)


def build_trading_calendar_from_csv(
    csv_path: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load trading calendar from a CSV that contains a single 'date' column
    listing all trading dates.

    Parameters
    ----------
    csv_path:
        Path to the trading calendar CSV.
    start_date / end_date:
        Optional date range filter (inclusive, 'YYYY-MM-DD' format).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Trading calendar CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Find the date column
    date_col = None
    for cand in ("date", "trade_date", "datetime", "calendar_date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # Try first column
        date_col = df.columns[0]
        logger.warning(
            "No recognised date column in calendar CSV; using first column '%s'", date_col
        )

    cal = pd.DataFrame()
    cal["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    cal = cal.dropna(subset=["date"])
    cal["is_open"] = True

    if start_date:
        cal = cal[cal["date"] >= pd.Timestamp(start_date)]
    if end_date:
        cal = cal[cal["date"] <= pd.Timestamp(end_date)]

    cal = cal.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    validate_trading_calendar(cal)

    logger.info(
        "trading_calendar: %d trading days (%s – %s)",
        len(cal),
        cal["date"].min().date(),
        cal["date"].max().date(),
    )
    return cal


def build_trading_calendar_from_ohlcv(
    ohlcv_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Derive trading calendar from unique dates in OHLCV data.
    This is a fallback when no calendar CSV is available.
    """
    dates = ohlcv_df["date"].dropna().unique()
    cal = pd.DataFrame({"date": pd.to_datetime(sorted(dates))})
    cal["is_open"] = True

    if start_date:
        cal = cal[cal["date"] >= pd.Timestamp(start_date)]
    if end_date:
        cal = cal[cal["date"] <= pd.Timestamp(end_date)]

    cal = cal.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    validate_trading_calendar(cal)

    logger.info(
        "trading_calendar (from OHLCV): %d trading days (%s – %s)",
        len(cal),
        cal["date"].min().date(),
        cal["date"].max().date(),
    )
    return cal
