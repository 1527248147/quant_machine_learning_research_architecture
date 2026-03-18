"""
Build instrument_master from normalised OHLCV data.

instrument_master schema:
  sid        : str  (unique stock identifier, e.g. '000001.SZ')
  symbol     : str  (display code; equals sid in this version)
  list_date  : Timestamp  (first date the stock appears in OHLCV)
  delist_date: Timestamp or NaT  (NaT if still active)
"""
from __future__ import annotations
import logging
from datetime import timedelta

import pandas as pd

from engine.core.constants import STILL_LISTED_GRACE_DAYS
from engine.schema.validators import validate_instrument_master

logger = logging.getLogger(__name__)


def build_instrument_master(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive instrument_master from normalised OHLCV long-table.

    Parameters
    ----------
    ohlcv_df:
        Normalised OHLCV DataFrame with columns [date, sid, ...].
        ``date`` must already be pd.Timestamp.

    Returns
    -------
    instrument_master DataFrame with columns [sid, symbol, list_date, delist_date].
    """
    if ohlcv_df.empty:
        raise ValueError("ohlcv_df is empty; cannot build instrument_master.")

    grouped = ohlcv_df.groupby("sid")["date"].agg(["min", "max"]).reset_index()
    grouped.columns = ["sid", "list_date", "last_date"]

    overall_max = grouped["last_date"].max()
    grace_cutoff = overall_max - timedelta(days=STILL_LISTED_GRACE_DAYS)

    # A stock is considered still listed if its last date is near the end
    grouped["delist_date"] = grouped["last_date"].where(
        grouped["last_date"] < grace_cutoff, other=pd.NaT
    )
    grouped["symbol"] = grouped["sid"]

    master = grouped[["sid", "symbol", "list_date", "delist_date"]].copy()
    master["list_date"] = pd.to_datetime(master["list_date"])
    master["delist_date"] = pd.to_datetime(master["delist_date"])

    validate_instrument_master(master)

    logger.info(
        "instrument_master: %d instruments, %d still listed, %d delisted",
        len(master),
        master["delist_date"].isna().sum(),
        master["delist_date"].notna().sum(),
    )
    return master
