"""
Core constants: enumerations and column name groups.
"""
from enum import Enum


# ---------------------------------------------------------------------------
# Sample state enumeration
# ---------------------------------------------------------------------------

class SampleState(str, Enum):
    NORMAL = "NORMAL"
    SUSPENDED = "SUSPENDED"
    MARKET_ONLY = "MARKET_ONLY"
    FACTOR_ONLY = "FACTOR_ONLY"
    NO_SOURCE_RECORD = "NO_SOURCE_RECORD"
    PARTIAL_FACTOR_MISSING = "PARTIAL_FACTOR_MISSING"
    INVALID_BASE_SAMPLE = "INVALID_BASE_SAMPLE"


class MarketState(str, Enum):
    OK = "OK"
    MISSING = "MISSING"
    SUSPENDED = "SUSPENDED"
    PARTIAL_MISSING = "PARTIAL_MISSING"


class FactorState(str, Enum):
    OK = "OK"
    ROW_MISSING = "ROW_MISSING"
    PARTIAL_MISSING = "PARTIAL_MISSING"
    ALL_MISSING = "ALL_MISSING"


# ---------------------------------------------------------------------------
# Column name groups
# ---------------------------------------------------------------------------

INDEX_COLS = ["date", "sid"]

# Raw OHLCV input aliases (detected in order)
RAW_DATE_CANDIDATES = ["date", "trade_date", "datetime"]
RAW_SID_CANDIDATES = ["symbol", "order_book_id", "instrument", "code", "ts_code"]

# Standardised market columns
MARKET_COLS = [
    "market.open",
    "market.high",
    "market.low",
    "market.close",
    "market.volume",
    "market.amount",
    "market.adj_factor",
]

# Required status columns in panel_base (12 columns)
REQUIRED_PANEL_BASE_STATUS_COLS = [
    "status.is_listed",
    "status.is_suspended",
    "status.has_market_record",
    "status.has_factor_record",
    "status.market_state",
    "status.factor_state",
    "status.bar_missing",
    "status.factor_row_missing",
    "status.factor_missing_ratio",
    "status.feature_all_missing",
    "status.sample_state",
    "status.sample_usable_for_feature",
]

# instrument_master required columns
INSTRUMENT_MASTER_COLS = ["sid", "symbol", "list_date", "delist_date"]

# trading_calendar required columns
TRADING_CALENDAR_COLS = ["date", "is_open"]

# daily_bars required columns
DAILY_BARS_REQUIRED_COLS = ["date", "sid"] + [
    "market.open", "market.high", "market.low", "market.close", "market.volume"
]

# factor_values required columns (excluding feature.* which are dynamic)
FACTOR_VALUES_BASE_COLS = ["date", "sid"]

# panel_base must NOT contain these
PANEL_BASE_FORBIDDEN_PREFIXES = ["label.", "label_valid.", "label_reason.", "pred."]

# Suspension threshold: volume == 0 means suspended
SUSPENSION_VOLUME_THRESHOLD = 0.0

# When deriving delist_date: if a stock's last date is within this many
# calendar days of the dataset's overall last date, treat as still listed.
STILL_LISTED_GRACE_DAYS = 90
