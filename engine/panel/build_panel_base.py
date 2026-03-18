"""
Assemble a panel_base chunk from active_session_index, daily_bars,
factor_values, status columns, and instrument_master.

panel_base is the unified bottom layer:
  - Row space is determined by active_session_index (never by OHLCV or factors alone).
  - daily_bars and factor_values are LEFT JOINed in.
  - status.* columns are pre-computed by status_builder.
  - panel_base must NOT contain any label.* columns.

Index: [date, sid]
"""
from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

from engine.schema.validators import validate_panel_base

logger = logging.getLogger(__name__)


def assemble_panel_chunk(
    active_session_index: pd.DataFrame,
    daily_bars: pd.DataFrame,
    factor_values: pd.DataFrame,
    status_df: pd.DataFrame,
    instrument_master: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Assemble one chunk (typically one year) of panel_base.

    Parameters
    ----------
    active_session_index:
        Columns: [date, sid] — the canonical row space for this chunk.
    daily_bars:
        Columns: [date, sid, market.*] — for this chunk's date range.
    factor_values:
        Columns: [date, sid, feature.*] — for this chunk's date range.
    status_df:
        DataFrame with index=[date, sid] and status.* columns
        (output of status_builder.build_status_columns).
    instrument_master:
        Optional. Columns: [sid, symbol, list_date, delist_date].
        Used to attach meta.* columns.

    Returns
    -------
    panel_base DataFrame with index=[date, sid] and all required columns.
    """
    base = active_session_index[["date", "sid"]].copy()

    # --- Left join market data ---
    mkt_cols = [c for c in daily_bars.columns if c.startswith("market.")]
    if mkt_cols:
        mkt = daily_bars[["date", "sid"] + mkt_cols]
        base = base.merge(mkt, on=["date", "sid"], how="left")

    # --- Left join factor data (force float32 to halve memory) ---
    feat_cols = [c for c in factor_values.columns if c.startswith("feature.")]
    if feat_cols:
        fac = factor_values[["date", "sid"] + feat_cols]
        base = base.merge(fac, on=["date", "sid"], how="left")
        for c in feat_cols:
            if c in base.columns:
                base[c] = base[c].astype("float32")

    # --- Left join meta data from instrument_master ---
    if instrument_master is not None:
        meta = instrument_master[["sid", "symbol", "list_date"]].copy()
        meta = meta.rename(columns={
            "symbol": "meta.symbol",
            "list_date": "meta.list_date",
        })
        base = base.merge(meta, on="sid", how="left")

    # --- Set index (active_session_index is already sorted, skip sort_index) ---
    base = base.set_index(["date", "sid"])

    # --- Attach status columns ---
    # status_df already has [date, sid] as index
    base = base.join(status_df, how="left")

    # --- Validate ---
    validate_panel_base(base)

    logger.info(
        "panel chunk: %d rows, %d columns (%d market, %d feature, %d status)",
        len(base),
        len(base.columns),
        len([c for c in base.columns if c.startswith("market.")]),
        len([c for c in base.columns if c.startswith("feature.")]),
        len([c for c in base.columns if c.startswith("status.")]),
    )
    return base
