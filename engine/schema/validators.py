"""
Schema validators for all core tables.

Each validator raises SchemaError with a descriptive message on failure.
"""
from __future__ import annotations
from typing import List, Optional
import pandas as pd

from engine.core.exceptions import SchemaError
from engine.core.constants import (
    REQUIRED_PANEL_BASE_STATUS_COLS,
    INSTRUMENT_MASTER_COLS,
    TRADING_CALENDAR_COLS,
    DAILY_BARS_REQUIRED_COLS,
    FACTOR_VALUES_BASE_COLS,
    PANEL_BASE_FORBIDDEN_PREFIXES,
)
from engine.schema.columns import has_forbidden_prefixes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_cols(df: pd.DataFrame, required: List[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"[{table_name}] Missing required columns: {missing}"
        )


def _require_unique_index(df: pd.DataFrame, keys: List[str], table_name: str) -> None:
    dupes = df.duplicated(subset=keys).sum()
    if dupes > 0:
        raise SchemaError(
            f"[{table_name}] ({', '.join(keys)}) is not unique: {dupes} duplicate rows"
        )


# ---------------------------------------------------------------------------
# Raw input validators
# ---------------------------------------------------------------------------

def validate_raw_ohlcv(df: pd.DataFrame) -> None:
    """Validate that a normalised OHLCV DataFrame has the minimum required columns."""
    _require_cols(df, ["date", "sid", "open", "high", "low", "close", "volume"], "raw_ohlcv")
    _require_unique_index(df, ["date", "sid"], "raw_ohlcv")


def validate_raw_factors(df: pd.DataFrame) -> None:
    """Validate that a normalised factor DataFrame has the minimum required columns."""
    _require_cols(df, ["date", "sid"], "raw_factors")
    _require_unique_index(df, ["date", "sid"], "raw_factors")
    feat_cols = [c for c in df.columns if c not in ("date", "sid")]
    if not feat_cols:
        raise SchemaError("[raw_factors] No feature columns found after metadata columns.")


# ---------------------------------------------------------------------------
# Asset table validators
# ---------------------------------------------------------------------------

def validate_instrument_master(df: pd.DataFrame) -> None:
    _require_cols(df, INSTRUMENT_MASTER_COLS, "instrument_master")
    _require_unique_index(df, ["sid"], "instrument_master")
    if df["list_date"].isna().any():
        raise SchemaError("[instrument_master] list_date must not contain NaT/NaN.")


def validate_trading_calendar(df: pd.DataFrame) -> None:
    _require_cols(df, TRADING_CALENDAR_COLS, "trading_calendar")
    _require_unique_index(df, ["date"], "trading_calendar")


def validate_daily_bars(df: pd.DataFrame) -> None:
    _require_cols(df, DAILY_BARS_REQUIRED_COLS, "daily_bars")
    _require_unique_index(df, ["date", "sid"], "daily_bars")


def validate_factor_values(df: pd.DataFrame) -> None:
    _require_cols(df, FACTOR_VALUES_BASE_COLS, "factor_values")
    _require_unique_index(df, ["date", "sid"], "factor_values")
    feat_cols = [c for c in df.columns if c.startswith("feature.")]
    if not feat_cols:
        raise SchemaError("[factor_values] No 'feature.*' columns found.")


# ---------------------------------------------------------------------------
# Panel validators
# ---------------------------------------------------------------------------

def validate_active_session_index(df: pd.DataFrame) -> None:
    _require_cols(df, ["date", "sid"], "active_session_index")
    _require_unique_index(df, ["date", "sid"], "active_session_index")


def validate_panel_base(df: pd.DataFrame) -> None:
    """Validate panel_base schema contract."""
    # Must have [date, sid] as index or columns
    idx_names = list(df.index.names)
    col_names = list(df.columns)
    all_names = idx_names + col_names

    for key in ("date", "sid"):
        if key not in all_names:
            raise SchemaError(
                f"[panel_base] '{key}' must be either in index or columns."
            )

    # Required status columns
    missing_status = [c for c in REQUIRED_PANEL_BASE_STATUS_COLS if c not in col_names]
    if missing_status:
        raise SchemaError(
            f"[panel_base] Missing required status columns: {missing_status}"
        )

    # Forbidden prefixes
    bad_cols = has_forbidden_prefixes(df)
    if bad_cols:
        raise SchemaError(
            f"[panel_base] Contains forbidden columns (labels must not appear in panel_base): {bad_cols}"
        )


def validate_target_block(df: pd.DataFrame, target_name: str) -> None:
    """Validate that a target_block has the required label columns."""
    required = [
        f"label.{target_name}",
        f"label_valid.{target_name}",
        f"label_reason.{target_name}",
    ]
    _require_cols(df, required, f"target_block[{target_name}]")
    idx_names = list(df.index.names)
    for key in ("date", "sid"):
        if key not in idx_names and key not in df.columns:
            raise SchemaError(
                f"[target_block] '{key}' must be present in index or columns."
            )


def validate_panel_labeled(panel_base: pd.DataFrame, panel_labeled: pd.DataFrame) -> None:
    """Validate that panel_labeled is a superset of panel_base with aligned index."""
    if not panel_base.index.equals(panel_labeled.index):
        raise SchemaError(
            "[panel_labeled] Index does not match panel_base index. "
            "They must be aligned."
        )
    missing = [c for c in panel_base.columns if c not in panel_labeled.columns]
    if missing:
        raise SchemaError(
            f"[panel_labeled] Missing columns from panel_base: {missing}"
        )
