"""
Build status columns for panel_base.

Takes the active_session_index as the canonical row space and derives all
12 required status.* fields by analysing join results with daily_bars,
factor_values, and instrument_master.

Suspension logic:
  In our data source, suspended stocks do NOT appear in the OHLCV table.
  Therefore: is_suspended = is_listed AND NOT has_market_record.
  Since every row in active_session_index is listed by construction,
  is_suspended ≡ ~has_market_record within the active row space.

All status derivations are fully vectorised (no row-wise apply).
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from engine.core.constants import (
    SampleState,
    MarketState,
    FactorState,
    REQUIRED_PANEL_BASE_STATUS_COLS,
)

logger = logging.getLogger(__name__)


def build_status_columns(
    active_session_index: pd.DataFrame,
    daily_bars: pd.DataFrame,
    factor_values: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all 12 status.* columns for every (date, sid) in active_session_index.

    Returns DataFrame with index=[date, sid] and all status.* columns.
    """
    idx = active_session_index[["date", "sid"]].copy()

    # -------------------------------------------------------------------
    # 1. has_market_record — does (date, sid) appear in daily_bars?
    # -------------------------------------------------------------------
    mkt_keys = daily_bars[["date", "sid"]].drop_duplicates()
    mkt_keys = mkt_keys.assign(_has_market=True)
    idx = idx.merge(mkt_keys, on=["date", "sid"], how="left")
    idx["status.has_market_record"] = idx["_has_market"].fillna(False)
    idx.drop(columns=["_has_market"], inplace=True)

    # -------------------------------------------------------------------
    # 2. has_factor_record — does (date, sid) appear in factor_values?
    # -------------------------------------------------------------------
    fac_keys = factor_values[["date", "sid"]].drop_duplicates()
    fac_keys = fac_keys.assign(_has_factor=True)
    idx = idx.merge(fac_keys, on=["date", "sid"], how="left")
    idx["status.has_factor_record"] = idx["_has_factor"].fillna(False)
    idx.drop(columns=["_has_factor"], inplace=True)

    # -------------------------------------------------------------------
    # 3. is_listed — always True in active_session_index by construction
    # -------------------------------------------------------------------
    idx["status.is_listed"] = True

    # -------------------------------------------------------------------
    # 4. is_suspended — listed but no OHLCV row on a trading day
    #    (suspended stocks are absent from the OHLCV source)
    # -------------------------------------------------------------------
    idx["status.is_suspended"] = ~idx["status.has_market_record"]

    # -------------------------------------------------------------------
    # 5. bar_missing — market record missing OR close price is NaN
    # -------------------------------------------------------------------
    if "market.close" in daily_bars.columns:
        close_df = daily_bars[["date", "sid", "market.close"]]
        idx = idx.merge(close_df, on=["date", "sid"], how="left")
        idx["status.bar_missing"] = (
            ~idx["status.has_market_record"] | idx["market.close"].isna()
        )
        idx.drop(columns=["market.close"], inplace=True)
    else:
        idx["status.bar_missing"] = ~idx["status.has_market_record"]

    # -------------------------------------------------------------------
    # 6. factor_row_missing — factor record entirely absent
    # -------------------------------------------------------------------
    idx["status.factor_row_missing"] = ~idx["status.has_factor_record"]

    # -------------------------------------------------------------------
    # 7. factor_missing_ratio — fraction of feature columns that are NaN
    # -------------------------------------------------------------------
    feat_cols = [c for c in factor_values.columns if c.startswith("feature.")]
    if feat_cols:
        fv_sub = factor_values[["date", "sid"]].copy()
        fv_sub["_fmr"] = (
            factor_values[feat_cols].isna().sum(axis=1) / len(feat_cols)
        ).astype("float32")
        idx = idx.merge(fv_sub, on=["date", "sid"], how="left")
        idx["status.factor_missing_ratio"] = idx["_fmr"].fillna(1.0).astype("float32")
        idx.drop(columns=["_fmr"], inplace=True)
    else:
        idx["status.factor_missing_ratio"] = np.float32(1.0)

    # -------------------------------------------------------------------
    # 8. feature_all_missing — all features NaN (or row missing)
    # -------------------------------------------------------------------
    idx["status.feature_all_missing"] = idx["status.factor_missing_ratio"] >= 1.0

    # -------------------------------------------------------------------
    # 9. market_state  (vectorised, no apply)
    # -------------------------------------------------------------------
    has_mkt = idx["status.has_market_record"]
    bar_missing = idx["status.bar_missing"]

    market_state = pd.Series(MarketState.OK.value, index=idx.index, dtype="object")
    # Suspended: no market record at all
    market_state = market_state.where(has_mkt, MarketState.SUSPENDED.value)
    # Partial missing: has record but close is NaN
    partial_mask = has_mkt & bar_missing
    market_state = market_state.where(~partial_mask, MarketState.PARTIAL_MISSING.value)
    idx["status.market_state"] = market_state.astype("category")

    # -------------------------------------------------------------------
    # 10. factor_state  (vectorised, no apply)
    # -------------------------------------------------------------------
    has_fac = idx["status.has_factor_record"]
    all_miss = idx["status.feature_all_missing"]
    fmr = idx["status.factor_missing_ratio"]

    factor_state = pd.Series(FactorState.OK.value, index=idx.index, dtype="object")
    factor_state = factor_state.where(has_fac, FactorState.ROW_MISSING.value)
    factor_state = factor_state.where(~(has_fac & all_miss), FactorState.ALL_MISSING.value)
    factor_state = factor_state.where(
        ~(has_fac & ~all_miss & (fmr > 0)), FactorState.PARTIAL_MISSING.value
    )
    idx["status.factor_state"] = factor_state.astype("category")

    # -------------------------------------------------------------------
    # 11. sample_state  (vectorised, no apply)
    # -------------------------------------------------------------------
    suspended = idx["status.is_suspended"]

    sample_state = pd.Series(SampleState.NORMAL.value, index=idx.index, dtype="object")
    # Order: assign from lowest priority to highest (last write wins)
    # PARTIAL_FACTOR_MISSING: both present but factor incomplete
    sample_state = sample_state.where(
        ~(has_mkt & has_fac & (fmr > 0)),
        SampleState.PARTIAL_FACTOR_MISSING.value,
    )
    # MARKET_ONLY: has market, no factor
    sample_state = sample_state.where(
        ~(has_mkt & ~has_fac),
        SampleState.MARKET_ONLY.value,
    )
    # FACTOR_ONLY: has factor, no market (= suspended with factor)
    sample_state = sample_state.where(
        ~(~has_mkt & has_fac),
        SampleState.FACTOR_ONLY.value,
    )
    # SUSPENDED: no market record (regardless of factor)
    # We override FACTOR_ONLY → SUSPENDED when the stock is suspended
    sample_state = sample_state.where(
        ~suspended,
        SampleState.SUSPENDED.value,
    )
    # NO_SOURCE_RECORD: neither source
    sample_state = sample_state.where(
        ~(~has_mkt & ~has_fac),
        SampleState.NO_SOURCE_RECORD.value,
    )
    idx["status.sample_state"] = sample_state.astype("category")

    # -------------------------------------------------------------------
    # 12. sample_usable_for_feature
    # -------------------------------------------------------------------
    idx["status.sample_usable_for_feature"] = (
        idx["status.has_factor_record"] & ~idx["status.feature_all_missing"]
    )

    # -------------------------------------------------------------------
    # Final: set index and verify
    # -------------------------------------------------------------------
    idx = idx.set_index(["date", "sid"]).sort_index()

    missing = [c for c in REQUIRED_PANEL_BASE_STATUS_COLS if c not in idx.columns]
    if missing:
        raise RuntimeError(f"status_builder is missing required columns: {missing}")

    logger.info(
        "status_builder: %d rows — sample_state distribution:\n%s",
        len(idx),
        idx["status.sample_state"].value_counts().to_string(),
    )
    return idx
