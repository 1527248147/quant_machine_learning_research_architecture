"""
Post-build sanity checks for panel_base.

These checks go beyond schema validation (validators.py handles that).
They verify that each status column has sensible, realistic distributions
given what we know about A-share data:
  - There MUST be suspended stocks (some stocks halt every year).
  - There MUST be both True and False values in boolean status columns.
  - factor_missing_ratio should span a range, not be all 0 or all 1.
  - sample_state should contain multiple distinct values.

Usage:
    from engine.panel.panel_validator import validate_panel_sanity
    issues = validate_panel_sanity(panel_base)
    # issues is a list of warning strings; empty = all checks passed.
"""
from __future__ import annotations
import logging
from typing import List

import pandas as pd

from engine.core.constants import SampleState, MarketState, FactorState

logger = logging.getLogger(__name__)


def validate_panel_sanity(panel: pd.DataFrame) -> List[str]:
    """
    Run sanity checks on a built panel_base. Returns a list of warning
    strings. An empty list means all checks passed.
    """
    issues: List[str] = []
    n = len(panel)
    if n == 0:
        issues.append("panel_base is empty (0 rows)")
        return issues

    # ------------------------------------------------------------------
    # 1. Boolean columns: must have both True and False
    # ------------------------------------------------------------------
    bool_cols = [
        "status.is_listed",
        "status.is_suspended",
        "status.has_market_record",
        "status.has_factor_record",
        "status.bar_missing",
        "status.factor_row_missing",
        "status.feature_all_missing",
        "status.sample_usable_for_feature",
    ]
    for col in bool_cols:
        if col not in panel.columns:
            issues.append(f"MISSING column: {col}")
            continue
        nunique = panel[col].nunique()
        if nunique < 2 and col != "status.is_listed":
            # is_listed is always True by construction, so skip that check
            only_val = panel[col].iloc[0] if nunique == 1 else "N/A"
            issues.append(
                f"SUSPICIOUS: {col} has only one unique value: {only_val} "
                f"(expected both True and False)"
            )

    # ------------------------------------------------------------------
    # 2. is_suspended: must have some True values
    #    A-share market always has suspended stocks
    # ------------------------------------------------------------------
    if "status.is_suspended" in panel.columns:
        n_suspended = panel["status.is_suspended"].sum()
        if n_suspended == 0:
            issues.append(
                "FAIL: status.is_suspended is all False — "
                "impossible for A-share data (suspended stocks are expected)"
            )
        else:
            pct = n_suspended / n * 100
            logger.info("is_suspended: %d rows (%.2f%%)", n_suspended, pct)
            if pct > 50:
                issues.append(
                    f"SUSPICIOUS: is_suspended rate = {pct:.1f}% — "
                    f"more than half the panel is suspended"
                )

    # ------------------------------------------------------------------
    # 3. has_market_record: majority should be True
    # ------------------------------------------------------------------
    if "status.has_market_record" in panel.columns:
        mkt_rate = panel["status.has_market_record"].mean()
        if mkt_rate < 0.5:
            issues.append(
                f"SUSPICIOUS: has_market_record rate = {mkt_rate*100:.1f}% — "
                f"less than half the panel has market data"
            )

    # ------------------------------------------------------------------
    # 4. factor_missing_ratio: should span a range
    # ------------------------------------------------------------------
    if "status.factor_missing_ratio" in panel.columns:
        fmr = panel["status.factor_missing_ratio"]
        if fmr.nunique() <= 1:
            issues.append(
                f"SUSPICIOUS: factor_missing_ratio has only one unique value: "
                f"{fmr.iloc[0]:.4f} (expected a range)"
            )

    # ------------------------------------------------------------------
    # 5. sample_state: should have multiple distinct values
    # ------------------------------------------------------------------
    if "status.sample_state" in panel.columns:
        states = set(panel["status.sample_state"].unique())
        if len(states) < 2:
            issues.append(
                f"SUSPICIOUS: sample_state has only {len(states)} distinct value(s): "
                f"{states} (expected multiple)"
            )
        # SUSPENDED must appear (A-share data always has suspensions)
        if SampleState.SUSPENDED.value not in states:
            issues.append(
                "FAIL: sample_state never contains SUSPENDED — "
                "impossible for A-share data"
            )

    # ------------------------------------------------------------------
    # 6. market_state: should have OK and at least one non-OK
    # ------------------------------------------------------------------
    if "status.market_state" in panel.columns:
        ms_values = set(panel["status.market_state"].unique())
        if MarketState.OK.value not in ms_values:
            issues.append("FAIL: market_state never contains OK")
        if len(ms_values) < 2:
            issues.append(
                f"SUSPICIOUS: market_state has only one value: {ms_values}"
            )

    # ------------------------------------------------------------------
    # 7. factor_state: should have OK
    # ------------------------------------------------------------------
    if "status.factor_state" in panel.columns:
        fs_values = set(panel["status.factor_state"].unique())
        if FactorState.OK.value not in fs_values:
            issues.append("FAIL: factor_state never contains OK")

    # ------------------------------------------------------------------
    # 8. sample_usable_for_feature: should have both True and False
    # ------------------------------------------------------------------
    if "status.sample_usable_for_feature" in panel.columns:
        usable_rate = panel["status.sample_usable_for_feature"].mean()
        if usable_rate >= 1.0:
            issues.append(
                "SUSPICIOUS: sample_usable_for_feature is all True "
                "(no unusable samples at all?)"
            )
        elif usable_rate <= 0.0:
            issues.append(
                "FAIL: sample_usable_for_feature is all False "
                "(no usable samples — nothing can be trained)"
            )

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------
    if issues:
        logger.warning(
            "panel_validator: %d issue(s) found:\n  %s",
            len(issues),
            "\n  ".join(issues),
        )
    else:
        logger.info("panel_validator: all sanity checks passed (%d rows)", n)

    return issues
