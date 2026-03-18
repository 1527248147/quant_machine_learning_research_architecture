"""
MomentumClsRecipe: 5-class momentum classification label.

Replicates the label_mom_cls logic from the LSTM-Momentum project.

Momentum line
-------------
For each (date t, sid), build a momentum line of LINE_LEN points:

    m_line[k] = close[t + k] - close[t - GAP_L]    for k = 0, 1, ..., LINE_LEN-1

where all offsets are in *trading days* (using the trading calendar).

Classification (vectorised)
---------------------------
  4 = Bounce   : momentum line crosses zero from negative to positive (exactly once)
  3 = Positive : all momentum line values > 0
  2 = Volatile : multiple zero-crossings, or mixed signals
  1 = Negative : all momentum line values < 0
  0 = Sink     : momentum line crosses zero from positive to negative (exactly once)

Invalid-label reasons
---------------------
- "insufficient_history"        : fewer than GAP_L trading days before t
- "insufficient_forward_window" : fewer than LINE_LEN-1 trading days after t
- "missing_close_in_window"     : any close price in the required window is NaN/<=0
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from engine.targets.base import BaseTargetRecipe


def _classify_momentum_lines_vectorised(m_lines: np.ndarray) -> np.ndarray:
    """
    Classify momentum lines in bulk.

    Parameters
    ----------
    m_lines : (N, LINE_LEN) array of momentum values.

    Returns
    -------
    (N,) int array of class labels (0-4).
    """
    signs = np.sign(m_lines)  # (N, LINE_LEN)

    # Count sign changes between consecutive columns
    # sign_changes[i, j] = 1 if signs[i, j+1] != signs[i, j] and both != 0
    diffs = np.diff(signs, axis=1)  # (N, LINE_LEN-1)

    # For counting crossings, we need to ignore zeros in signs.
    # Approach: for each row, find first nonzero and last nonzero sign,
    # and count how many times sign flips among nonzero elements.
    #
    # Efficient approximation: count columns where diff != 0 AND both
    # neighbouring signs are nonzero.
    left_signs = signs[:, :-1]
    right_signs = signs[:, 1:]
    both_nonzero = (left_signs != 0) & (right_signs != 0)
    crossings = np.sum((diffs != 0) & both_nonzero, axis=1)  # (N,)

    # Determine first and last nonzero sign per row
    # If a row is all-zero, we call it Volatile (2)
    has_positive = np.any(signs > 0, axis=1)
    has_negative = np.any(signs < 0, axis=1)
    all_zero = ~has_positive & ~has_negative

    # For first/last nonzero sign, scan from left/right
    # Use argmax on abs(signs) to find first nonzero
    abs_signs = np.abs(signs)
    first_nz_idx = np.argmax(abs_signs, axis=1)  # first nonzero index
    last_nz_idx = abs_signs.shape[1] - 1 - np.argmax(abs_signs[:, ::-1], axis=1)

    n = len(m_lines)
    row_idx = np.arange(n)
    first_sign = signs[row_idx, first_nz_idx]
    last_sign = signs[row_idx, last_nz_idx]

    # Default = Volatile (2)
    result = np.full(n, 2, dtype=np.int16)

    # No crossings
    no_cross = crossings == 0
    result[no_cross & (first_sign > 0)] = 3   # Positive
    result[no_cross & (first_sign < 0)] = 1   # Negative

    # Single crossing
    one_cross = crossings == 1
    result[one_cross & (first_sign < 0) & (last_sign > 0)] = 4  # Bounce
    result[one_cross & (first_sign > 0) & (last_sign < 0)] = 0  # Sink

    # Multiple crossings → already 2 (Volatile)
    # All zero → already 2
    result[all_zero] = 2

    return result


class MomentumClsRecipe(BaseTargetRecipe):
    """5-class momentum classification recipe."""

    def __init__(
        self,
        name: str,
        gap: int = 4,
        line_len: int = 7,
        price_col: str = "market.close",
        trading_calendar: pd.DataFrame | None = None,
        **kwargs,
    ):
        self.name = name
        self.gap = gap
        self.line_len = line_len
        self.price_col = price_col
        self.trading_calendar = trading_calendar

    def required_columns(self) -> List[str]:
        return [self.price_col]

    def class_names(self) -> Dict[int, str]:
        return {
            0: "Sink (正→负)",
            1: "Negative (全负)",
            2: "Volatile (震荡)",
            3: "Positive (全正)",
            4: "Bounce (负→正)",
        }

    def compute(self, panel_base: pd.DataFrame) -> pd.DataFrame:
        GAP = self.gap
        LINE_LEN = self.line_len
        label_col = f"label.{self.name}"
        valid_col = f"label_valid.{self.name}"
        reason_col = f"label_reason.{self.name}"

        # --- 1. Extract only needed data (no full copy) ---------------
        idx = panel_base.index  # [date, sid] MultiIndex
        close_vals = panel_base[self.price_col].values.astype(np.float64)
        dates = idx.get_level_values("date")
        sids = idx.get_level_values("sid")

        # --- 2. Build trading-date array ------------------------------
        if self.trading_calendar is not None:
            if "date" in self.trading_calendar.columns:
                cal_dates = self.trading_calendar["date"]
            else:
                cal_dates = self.trading_calendar.index.to_series()
            trade_dates = cal_dates.drop_duplicates().sort_values().values
        else:
            trade_dates = np.sort(dates.unique())

        n_td = len(trade_dates)

        # Vectorised date → integer index mapping
        date_rank = pd.Series(
            np.arange(n_td, dtype=np.int32), index=trade_dates
        )
        row_date_idx = date_rank.reindex(dates.values).values  # (n_rows,) int or NaN

        # Handle dates not in calendar (shouldn't happen, but be safe)
        not_in_cal = np.isnan(row_date_idx)
        row_date_idx = np.where(not_in_cal, -1, row_date_idx).astype(np.int32)

        n_rows = len(close_vals)

        # --- 3. Structural validity -----------------------------------
        has_history = row_date_idx >= GAP
        has_forward = (row_date_idx + LINE_LEN - 1) < n_td
        in_calendar = row_date_idx >= 0
        structurally_valid = in_calendar & has_history & has_forward

        # Reason for structurally invalid
        reasons = np.full(n_rows, "", dtype=object)
        reasons[~in_calendar] = "not_in_calendar"
        reasons[in_calendar & ~has_history] = "insufficient_history"
        reasons[in_calendar & has_history & ~has_forward] = "insufficient_forward_window"

        labels = np.full(n_rows, np.nan, dtype=np.float32)
        valid_flags = np.zeros(n_rows, dtype=bool)

        valid_indices = np.where(structurally_valid)[0]
        n_valid = len(valid_indices)

        if n_valid > 0:
            # --- 4. Gather close prices via merge for each offset -----
            # Offsets needed: -GAP, 0, 1, ..., LINE_LEN-1
            offsets = [-GAP] + list(range(LINE_LEN))
            n_offsets = len(offsets)

            # Build a (date, sid) → close lookup (fast hash-based)
            close_df = pd.DataFrame({
                "date": dates.values,
                "sid": sids.values,
                "close": close_vals,
            })
            # Use merge instead of MultiIndex reindex: much faster
            close_ref = close_df.rename(columns={"date": "_lookup_date", "close": "_close"})

            valid_date_idx = row_date_idx[valid_indices]
            valid_sids = sids.values[valid_indices]

            # Pre-allocate close matrix
            close_matrix = np.full((n_valid, n_offsets), np.nan, dtype=np.float64)

            for j, offset in enumerate(offsets):
                target_td_idx = valid_date_idx + offset
                # Clip to valid range (already checked structurally, but be safe)
                target_td_idx = np.clip(target_td_idx, 0, n_td - 1)
                lookup_dates = trade_dates[target_td_idx]

                # Build lookup frame and merge
                query = pd.DataFrame({
                    "_row": np.arange(n_valid),
                    "_lookup_date": lookup_dates,
                    "sid": valid_sids,
                })
                merged = query.merge(close_ref, on=["_lookup_date", "sid"], how="left")
                # Sort back by _row to maintain order
                merged = merged.sort_values("_row")
                close_matrix[:, j] = merged["_close"].values

            # --- 5. Check for missing prices --------------------------
            any_bad = np.isnan(close_matrix).any(axis=1) | (close_matrix <= 0).any(axis=1)

            # --- 6. Compute momentum lines ----------------------------
            base_close = close_matrix[:, 0]          # close at t-GAP
            forward_close = close_matrix[:, 1:]      # close at t, t+1, ..., t+LINE_LEN-1
            m_lines = forward_close - base_close[:, np.newaxis]

            # --- 7. Vectorised classification -------------------------
            # Only classify rows without missing prices
            good_mask = ~any_bad
            good_indices = np.where(good_mask)[0]

            if len(good_indices) > 0:
                cls = _classify_momentum_lines_vectorised(m_lines[good_indices])
                # Write back
                global_good = valid_indices[good_indices]
                labels[global_good] = cls.astype(np.float32)
                valid_flags[global_good] = True

            # Mark missing-price rows
            bad_global = valid_indices[np.where(any_bad)[0]]
            reasons[bad_global] = "missing_close_in_window"

        # --- 8. Assemble target_block ---------------------------------
        target_block = pd.DataFrame(
            {
                label_col: labels,
                valid_col: valid_flags,
                reason_col: reasons,
            },
            index=idx,
        )
        return target_block
