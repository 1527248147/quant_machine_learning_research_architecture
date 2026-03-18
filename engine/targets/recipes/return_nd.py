"""
ReturnRecipe: compute forward returns with flexible entry/exit price points.

Supports standard quant-style notation for price combinations:

    formula = "o1c2"   →  return = (close_day2 - open_day1) / open_day1

General pattern: {entry_type}{entry_day}{exit_type}{exit_day}
    - entry_type / exit_type:  'o' = open,  'c' = close
    - entry_day / exit_day:    trading-day offset from t (0 = today)

Common examples
---------------
    "c0c1"   close-to-close 1-day return  (最常用的日收益)
    "o1c1"   open-to-close intraday       (日内收益)
    "c0o1"   overnight return             (隔夜收益)
    "o1c2"   next-day open to day-after close  (持仓 2 天)
    "o1c5"   next-day open to 5th-day close    (持仓约 1 周)

Invalid-label reasons
---------------------
- "insufficient_history"    : entry_day requires looking back but not enough dates
- "insufficient_forward"    : exit_day goes beyond available trading dates
- "missing_entry_price"     : entry price is NaN or <= 0
- "missing_exit_price"      : exit price is NaN or <= 0
"""
from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd

from engine.targets.base import BaseTargetRecipe


def _parse_formula(formula: str):
    """
    Parse a formula like "o1c2" into (entry_type, entry_day, exit_type, exit_day).

    Returns
    -------
    (entry_type, entry_day, exit_type, exit_day) : (str, int, str, int)
        entry_type / exit_type in {'o', 'c'}
        entry_day / exit_day are non-negative integers
    """
    m = re.fullmatch(r"([oc])(\d+)([oc])(\d+)", formula.strip().lower())
    if not m:
        raise ValueError(
            f"Invalid return formula '{formula}'. "
            f"Expected pattern like 'o1c2', 'c0c1', etc. "
            f"Format: {{o|c}}{{day}}{{o|c}}{{day}}"
        )
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))


def _price_col_for_type(price_type: str, open_col: str, close_col: str) -> str:
    """Map 'o'/'c' to the actual column name."""
    return open_col if price_type == "o" else close_col


class ReturnRecipe(BaseTargetRecipe):
    """
    Flexible forward return recipe with o/c entry/exit notation.

    Parameters
    ----------
    name : str
        Target name (e.g. "return_o1c2").
    formula : str
        Price formula like "o1c2", "c0c1", etc.
    open_col : str
        Column name for open price in panel_base.
    close_col : str
        Column name for close price in panel_base.
    trading_calendar : DataFrame, optional
        Trading calendar for date alignment.
    window : int, optional
        **Backward compatibility**: if formula is not given, treat as c0c{window}.
    price_col : str, optional
        **Backward compatibility**: if formula is not given, this sets close_col.
    """

    def __init__(
        self,
        name: str,
        formula: str | None = None,
        open_col: str = "market.open",
        close_col: str = "market.close",
        trading_calendar: pd.DataFrame | None = None,
        # backward compatibility with old config
        window: int | None = None,
        price_col: str | None = None,
        **kwargs,
    ):
        self.name = name
        self.trading_calendar = trading_calendar

        # Resolve formula: explicit formula takes priority
        if formula is not None:
            self.entry_type, self.entry_day, self.exit_type, self.exit_day = (
                _parse_formula(formula)
            )
        elif window is not None:
            # backward compat: window=N → c0cN
            self.entry_type, self.entry_day = "c", 0
            self.exit_type, self.exit_day = "c", window
        else:
            raise ValueError(
                "ReturnRecipe requires either 'formula' (e.g. 'o1c2') "
                "or 'window' (backward compat, e.g. window=1 → c0c1)."
            )

        # Resolve column names
        if price_col is not None:
            # backward compat: price_col sets close_col
            close_col = price_col
        self.open_col = open_col
        self.close_col = close_col

        self.entry_col = _price_col_for_type(self.entry_type, open_col, close_col)
        self.exit_col = _price_col_for_type(self.exit_type, open_col, close_col)

    @property
    def formula_str(self) -> str:
        return f"{self.entry_type}{self.entry_day}{self.exit_type}{self.exit_day}"

    def required_columns(self) -> List[str]:
        cols = {self.entry_col, self.exit_col}
        return list(cols)

    def compute(self, panel_base: pd.DataFrame) -> pd.DataFrame:
        label_col = f"label.{self.name}"
        valid_col = f"label_valid.{self.name}"
        reason_col = f"label_reason.{self.name}"

        idx = panel_base.index  # [date, sid] MultiIndex
        dates = idx.get_level_values("date")
        sids = idx.get_level_values("sid")

        # --- 1. Build trading calendar index --------------------------------
        if self.trading_calendar is not None:
            if "date" in self.trading_calendar.columns:
                cal_dates = self.trading_calendar["date"]
            else:
                cal_dates = self.trading_calendar.index.to_series()
            trade_dates = cal_dates.drop_duplicates().sort_values().values
        else:
            trade_dates = np.sort(dates.unique())

        n_td = len(trade_dates)

        # date → integer index
        date_rank = pd.Series(np.arange(n_td, dtype=np.int32), index=trade_dates)
        row_date_idx = date_rank.reindex(dates.values).values
        not_in_cal = np.isnan(row_date_idx)
        row_date_idx = np.where(not_in_cal, -1, row_date_idx).astype(np.int32)

        n_rows = len(dates)

        # --- 2. Compute entry/exit date indices -----------------------------
        entry_td_idx = row_date_idx + self.entry_day
        exit_td_idx = row_date_idx + self.exit_day

        in_calendar = row_date_idx >= 0
        entry_in_range = (entry_td_idx >= 0) & (entry_td_idx < n_td)
        exit_in_range = (exit_td_idx >= 0) & (exit_td_idx < n_td)
        structurally_valid = in_calendar & entry_in_range & exit_in_range

        # --- 3. Look up entry and exit prices via merge ---------------------
        entry_prices = np.full(n_rows, np.nan, dtype=np.float64)
        exit_prices = np.full(n_rows, np.nan, dtype=np.float64)

        valid_mask = structurally_valid
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            valid_sids = sids.values[valid_indices]

            # Build reference lookup: (date, sid) → price
            # We need two lookups: one for entry price column, one for exit
            entry_ref = pd.DataFrame({
                "_lookup_date": dates.values,
                "sid": sids.values,
                "_price": panel_base[self.entry_col].values.astype(np.float64),
            })
            exit_ref = pd.DataFrame({
                "_lookup_date": dates.values,
                "sid": sids.values,
                "_price": panel_base[self.exit_col].values.astype(np.float64),
            })

            # Entry prices
            entry_lookup_dates = trade_dates[
                np.clip(entry_td_idx[valid_indices], 0, n_td - 1)
            ]
            query_entry = pd.DataFrame({
                "_row": np.arange(len(valid_indices)),
                "_lookup_date": entry_lookup_dates,
                "sid": valid_sids,
            })
            merged_entry = query_entry.merge(entry_ref, on=["_lookup_date", "sid"], how="left")
            merged_entry = merged_entry.sort_values("_row")
            entry_prices[valid_indices] = merged_entry["_price"].values

            # Exit prices
            exit_lookup_dates = trade_dates[
                np.clip(exit_td_idx[valid_indices], 0, n_td - 1)
            ]
            query_exit = pd.DataFrame({
                "_row": np.arange(len(valid_indices)),
                "_lookup_date": exit_lookup_dates,
                "sid": valid_sids,
            })
            merged_exit = query_exit.merge(exit_ref, on=["_lookup_date", "sid"], how="left")
            merged_exit = merged_exit.sort_values("_row")
            exit_prices[valid_indices] = merged_exit["_price"].values

        # --- 4. Compute return and validity ---------------------------------
        label = (exit_prices - entry_prices) / entry_prices

        bad_entry = np.isnan(entry_prices) | (entry_prices <= 0)
        bad_exit = np.isnan(exit_prices) | (exit_prices <= 0)

        valid = structurally_valid & ~bad_entry & ~bad_exit

        reason = np.full(n_rows, "", dtype=object)
        reason[~in_calendar] = "not_in_calendar"
        reason[in_calendar & ~entry_in_range] = "insufficient_history"
        reason[in_calendar & entry_in_range & ~exit_in_range] = "insufficient_forward"
        reason[structurally_valid & bad_entry] = "missing_entry_price"
        reason[structurally_valid & ~bad_entry & bad_exit] = "missing_exit_price"

        label[~valid] = np.nan

        # --- 5. Assemble target_block --------------------------------------
        target_block = pd.DataFrame(
            {
                label_col: label.astype(np.float32),
                valid_col: valid,
                reason_col: reason,
            },
            index=idx,
        )
        return target_block


# Backward-compatible alias
ReturnNdRecipe = ReturnRecipe
