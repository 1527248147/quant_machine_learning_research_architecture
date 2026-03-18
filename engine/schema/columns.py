"""
Column name helpers and prefix utilities.
"""
from __future__ import annotations
from typing import List
import pandas as pd


FEATURE_PREFIX = "feature."
MARKET_PREFIX = "market."
META_PREFIX = "meta."
STATUS_PREFIX = "status."
LABEL_PREFIX = "label."
LABEL_VALID_PREFIX = "label_valid."
LABEL_REASON_PREFIX = "label_reason."
PRED_PREFIX = "pred."


def feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def market_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(MARKET_PREFIX)]


def status_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(STATUS_PREFIX)]


def label_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(LABEL_PREFIX)]


def has_forbidden_prefixes(df: pd.DataFrame) -> List[str]:
    """Return any columns that must not appear in panel_base."""
    forbidden = [LABEL_PREFIX, LABEL_VALID_PREFIX, LABEL_REASON_PREFIX, PRED_PREFIX]
    return [c for c in df.columns for p in forbidden if c.startswith(p)]
