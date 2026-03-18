"""
Build active_session_index: the logical row space for panel_base.

active_session_index answers "which (date, sid) samples should exist?"

Rules:
  - Include all (date, sid) where trading_calendar.is_open == True
    AND date >= instrument_master.list_date
    AND (instrument_master.delist_date is NaT OR date <= delist_date)
  - Suspended and bar-missing rows ARE included (status flags handle that).
  - Rows for stocks not yet listed or already delisted are excluded.
"""
from __future__ import annotations
import logging

import pandas as pd

from engine.schema.validators import validate_active_session_index

logger = logging.getLogger(__name__)


def build_active_session_index(
    instrument_master: pd.DataFrame,
    trading_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute active_session_index as a cross-join filtered by listing window.

    Parameters
    ----------
    instrument_master:
        Columns: [sid, symbol, list_date, delist_date].
    trading_calendar:
        Columns: [date, is_open]. Only open days should be in this table.

    Returns
    -------
    DataFrame with columns [date, sid], sorted by [date, sid].
    """
    open_dates = trading_calendar[trading_calendar["is_open"]]["date"].values
    open_dates_ts = pd.to_datetime(open_dates)

    # Cross-join: one row per (date, sid) combination
    # Use merge with a temporary key for an efficient cross-join
    cal_df = pd.DataFrame({"date": open_dates_ts, "_key": 1})
    inst_df = instrument_master[["sid", "list_date", "delist_date"]].copy()
    inst_df["_key"] = 1

    crossed = pd.merge(cal_df, inst_df, on="_key").drop(columns="_key")

    # Filter by listing window
    in_window = crossed["date"] >= crossed["list_date"]

    has_delist = crossed["delist_date"].notna()
    past_delist = has_delist & (crossed["date"] > crossed["delist_date"])

    active = crossed[in_window & ~past_delist][["date", "sid"]].copy()
    active = active.sort_values(["date", "sid"]).reset_index(drop=True)

    validate_active_session_index(active)

    logger.info(
        "active_session_index: %d rows (%d dates × ~%d sids avg)",
        len(active),
        active["date"].nunique(),
        len(active) // max(active["date"].nunique(), 1),
    )
    return active
