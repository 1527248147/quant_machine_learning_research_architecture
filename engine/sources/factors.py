"""
Factor source: reads raw yearly parquet files and produces a clean long-table
with columns [date, sid, feature.<name>, …].

Handles the specific format of alpha158_plus_fund_yearly_parquet:
  - datetime column  : datetime.date objects
  - order_book_id    : 'XXXXXX.XSHG' / 'XXXXXX.XSHE'  →  'XXXXXX.SH' / 'XXXXXX.SZ'
  - instrument       : 'SHxxxxxx' / 'SZxxxxxx'  (alternative id, dropped after normalisation)
  - 473 numeric factor columns  →  renamed to 'feature.<original_name>'
"""
from __future__ import annotations
import gc
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd

from engine.core.exceptions import SourceError
from engine.io.parquet_io import iter_yearly_parquets, load_yearly_parquets
from engine.schema.validators import validate_raw_factors

logger = logging.getLogger(__name__)

# Columns that are metadata, not features
_META_COLS = {"datetime", "date", "instrument", "order_book_id", "sid", "symbol",
              "trade_date", "ts_code", "code"}

# Exchange suffix normalisation (shared with OHLCV source)
_EXCHANGE_MAP: dict[str, str] = {
    "XSHE": "SZ",
    "XSHG": "SH",
    "xshe": "SZ",
    "xshg": "SH",
}


def _normalise_symbol(s: pd.Series) -> pd.Series:
    def _convert(sym: str) -> str:
        sym = str(sym).strip()
        if "." in sym:
            code, exch = sym.rsplit(".", 1)
            mapped = _EXCHANGE_MAP.get(exch.upper(), exch.upper())
            return f"{code}.{mapped}"
        if sym[:2].upper() in ("SH", "SZ", "BJ"):
            prefix = sym[:2].upper()
            code = sym[2:]
            return f"{code}.{prefix}"
        if sym.isdigit() and len(sym) == 6:
            if sym[0] in ("6", "9"):
                return f"{sym}.SH"
            elif sym[0] in ("4", "8"):
                return f"{sym}.BJ"
            else:
                return f"{sym}.SZ"
        return sym
    return s.map(_convert)


def _normalise_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


class FactorSource:
    """
    Reads raw factor yearly parquets and produces a standardised long-table.

    Output columns: date (Timestamp), sid (str), feature.<name> (floats).
    The 'feature.' prefix is prepended to all non-metadata numeric columns.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        exclude_cols: Optional[List[str]] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.start_year = start_year
        self.end_year = end_year
        self.exclude_cols: List[str] = exclude_cols or []

    def load(self) -> pd.DataFrame:
        """Load and normalise raw factor data."""
        df = load_yearly_parquets(
            self.raw_dir,
            start_year=self.start_year,
            end_year=self.end_year,
        )
        df = self._normalise(df)
        validate_raw_factors(df)
        feat_cols = [c for c in df.columns if c.startswith("feature.")]
        logger.info(
            "FactorSource: %d rows, %d unique dates, %d unique sids, %d features",
            len(df),
            df["date"].nunique(),
            df["sid"].nunique(),
            len(feat_cols),
        )
        return df

    def iter_years(self) -> Iterator[Tuple[int, pd.DataFrame]]:
        """Yield (year, normalised_df) one year at a time."""
        for year, raw_df in iter_yearly_parquets(self.raw_dir, self.start_year, self.end_year):
            normalised = self._normalise(raw_df)
            del raw_df
            yield year, normalised

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        # Resolve date column
        date_col = None
        for cand in ("datetime", "date", "trade_date"):
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None:
            raise SourceError(
                f"Factors: cannot find a date column. Available: {list(df.columns)}"
            )
        if date_col != "date":
            df = df.rename(columns={date_col: "date"})

        # Resolve sid column — prefer order_book_id, then instrument
        sid_col = None
        for cand in ("order_book_id", "instrument", "symbol", "sid", "code", "ts_code"):
            if cand in df.columns:
                sid_col = cand
                break
        if sid_col is None:
            raise SourceError(
                f"Factors: cannot find a sid/symbol column. Available: {list(df.columns)}"
            )
        if sid_col != "sid":
            df = df.rename(columns={sid_col: "sid"})
        # Drop the other id column if it still exists
        for extra in ("instrument", "order_book_id", "symbol", "code", "ts_code"):
            if extra in df.columns:
                df = df.drop(columns=[extra])

        # Normalise date and sid
        df["date"] = _normalise_date(df["date"])
        df["sid"] = _normalise_symbol(df["sid"])

        # Identify feature columns (all numeric non-metadata cols)
        feature_cols_raw = [
            c for c in df.columns
            if c not in _META_COLS and c not in self.exclude_cols
        ]
        if not feature_cols_raw:
            raise SourceError("Factors: no feature columns detected after excluding metadata.")

        # Rename to feature.* prefix
        rename_map = {c: f"feature.{c}" for c in feature_cols_raw
                      if not c.startswith("feature.")}
        df = df.rename(columns=rename_map)

        # Drop rows with null date or sid
        n_before = len(df)
        df = df.dropna(subset=["date", "sid"])
        if len(df) < n_before:
            logger.warning(
                "Factors: dropped %d rows with null date or sid", n_before - len(df)
            )

        # Deduplicate on (date, sid) — keep first
        n_before = len(df)
        df = df.drop_duplicates(subset=["date", "sid"], keep="first")
        if len(df) < n_before:
            logger.warning(
                "Factors: dropped %d duplicate (date, sid) rows", n_before - len(df)
            )

        df = df.sort_values(["date", "sid"]).reset_index(drop=True)

        # Reorder: date, sid, feature.*
        feat_cols = [c for c in df.columns if c.startswith("feature.")]
        return df[["date", "sid"] + feat_cols]
