"""
OHLCV source: reads raw yearly parquet files and produces a clean long-table
with columns [date, sid, open, high, low, close, volume, amount, adj_factor].

Handles the specific format of rq_ohlcv_yearly_parquet2005-2025:
  - date column  : string 'YYYY-MM-DD'
  - symbol column: 'XXXXXX.XSHE' or 'XXXXXX.XSHG'  →  'XXXXXX.SZ' / 'XXXXXX.SH'
  - money column : renamed to 'amount'
  - factor column: adjustment factor, renamed to 'adj_factor'
"""
from __future__ import annotations
import gc
import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

import pandas as pd

from engine.core.exceptions import SourceError
from engine.io.parquet_io import iter_yearly_parquets, load_yearly_parquets
from engine.schema.validators import validate_raw_ohlcv

logger = logging.getLogger(__name__)

# Mapping from raw OHLCV column name to standardised name
_OHLCV_RENAME: dict[str, str] = {
    "money": "amount",
    "factor": "adj_factor",
    # Fallback aliases
    "trade_date": "date",
    "datetime": "date",
    "order_book_id": "sid",
    "instrument": "sid",
    "code": "sid",
    "ts_code": "sid",
}

# Exchange suffix normalisation map
_EXCHANGE_MAP: dict[str, str] = {
    "XSHE": "SZ",
    "XSHG": "SH",
    "xshe": "SZ",
    "xshg": "SH",
}


def _normalise_symbol(s: pd.Series) -> pd.Series:
    """
    Normalise stock symbols to 'XXXXXX.SZ' / 'XXXXXX.SH' / 'XXXXXX.BJ'.

    Handles:
      '002017.XSHE'  →  '002017.SZ'
      '600000.XSHG'  →  '600000.SH'
      'SH600000'     →  '600000.SH'
      'SZ000001'     →  '000001.SZ'
    """
    def _convert(sym: str) -> str:
        sym = str(sym).strip()

        # Format: XXXXXX.EXCHANGE
        if "." in sym:
            code, exch = sym.rsplit(".", 1)
            mapped = _EXCHANGE_MAP.get(exch.upper(), exch.upper())
            # Map BJ / XBEJ
            if mapped in ("BJ", "XBEJ"):
                mapped = "BJ"
            return f"{code}.{mapped}"

        # Format: SHxxxxxx / SZxxxxxx / BJxxxxxx
        if sym[:2].upper() in ("SH", "SZ", "BJ"):
            prefix = sym[:2].upper()
            code = sym[2:]
            return f"{code}.{prefix}"

        # 6-digit plain code: infer exchange
        if sym.isdigit() and len(sym) == 6:
            if sym[0] in ("6", "9"):
                return f"{sym}.SH"
            elif sym[0] in ("4", "8"):
                return f"{sym}.BJ"
            else:
                return f"{sym}.SZ"

        return sym  # fallback: return as-is

    return s.map(_convert)


def _normalise_date(s: pd.Series) -> pd.Series:
    """Convert various date representations to pd.Timestamp (date precision)."""
    return pd.to_datetime(s, errors="coerce").dt.normalize()


class OHLCVSource:
    """
    Reads raw OHLCV yearly parquets and produces a standardised long-table.

    Output columns: date (Timestamp), sid (str), open, high, low, close,
                    volume, amount, adj_factor (floats).
    """

    def __init__(
        self,
        raw_dir: str | Path,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.start_year = start_year
        self.end_year = end_year

    def load(self) -> pd.DataFrame:
        """Load and normalise all OHLCV years into one DataFrame."""
        df = load_yearly_parquets(
            self.raw_dir,
            start_year=self.start_year,
            end_year=self.end_year,
        )
        df = self._normalise(df)
        validate_raw_ohlcv(df)
        logger.info(
            "OHLCVSource: %d rows, %d unique dates, %d unique symbols",
            len(df),
            df["date"].nunique(),
            df["sid"].nunique(),
        )
        return df

    def load_lightweight(self) -> pd.DataFrame:
        """Load only [date, sid] from all OHLCV years. Used for metadata building."""
        frames = []
        for _, raw_df in iter_yearly_parquets(self.raw_dir, self.start_year, self.end_year):
            norm = self._normalise(raw_df)
            frames.append(norm[["date", "sid"]])
            del raw_df, norm
        result = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()
        result = result.drop_duplicates(subset=["date", "sid"], keep="first")
        logger.info(
            "OHLCVSource (lightweight): %d rows, %d dates, %d sids",
            len(result), result["date"].nunique(), result["sid"].nunique(),
        )
        return result

    def iter_years(self) -> Iterator[Tuple[int, pd.DataFrame]]:
        """Yield (year, normalised_df) one year at a time."""
        for year, raw_df in iter_yearly_parquets(self.raw_dir, self.start_year, self.end_year):
            normalised = self._normalise(raw_df)
            del raw_df
            yield year, normalised

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rename known aliases first
        rename_map = {k: v for k, v in _OHLCV_RENAME.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Detect date column
        if "date" not in df.columns:
            raise SourceError(
                f"OHLCV: cannot find a date column. Available: {list(df.columns)}"
            )

        # Detect sid column
        if "symbol" in df.columns and "sid" not in df.columns:
            df = df.rename(columns={"symbol": "sid"})
        if "sid" not in df.columns:
            raise SourceError(
                f"OHLCV: cannot find a sid/symbol column. Available: {list(df.columns)}"
            )

        # Normalise date and sid
        df["date"] = _normalise_date(df["date"])
        df["sid"] = _normalise_symbol(df["sid"])

        # Ensure required numeric columns exist
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise SourceError(f"OHLCV: required column '{col}' not found.")

        # Fill missing optional columns
        if "amount" not in df.columns:
            df["amount"] = float("nan")
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0

        # Drop rows where date or sid is null
        n_before = len(df)
        df = df.dropna(subset=["date", "sid"])
        if len(df) < n_before:
            logger.warning(
                "OHLCV: dropped %d rows with null date or sid", n_before - len(df)
            )

        # Deduplicate on (date, sid) — keep first
        n_before = len(df)
        df = df.drop_duplicates(subset=["date", "sid"], keep="first")
        if len(df) < n_before:
            logger.warning(
                "OHLCV: dropped %d duplicate (date, sid) rows", n_before - len(df)
            )

        # Sort
        df = df.sort_values(["date", "sid"]).reset_index(drop=True)

        return df[["date", "sid", "open", "high", "low", "close", "volume", "amount", "adj_factor"]]
