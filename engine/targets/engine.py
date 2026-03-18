"""
Target engine: orchestrate building a target_block from panel_base.

Usage:
    # From an already-loaded DataFrame:
    from engine.targets.engine import build_target_block
    target_block = build_target_block(panel_base, "return_5d")

    # From a parquet path (memory-efficient — only loads required columns):
    from engine.targets.engine import build_target_block_from_path
    target_block = build_target_block_from_path(
        "data/processed/panel/panel_base.parquet", "return_5d",
        trading_calendar=cal,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from engine.core.exceptions import TargetBuildError
from engine.schema.validators import validate_target_block
from engine.targets.registry import get_recipe, get_spec

logger = logging.getLogger(__name__)


def _make_recipe(target_name: str, trading_calendar=None, **recipe_kwargs):
    """Create a recipe instance with merged spec.params + overrides."""
    spec = get_spec(target_name)
    params = {**spec.params, **recipe_kwargs}
    params["name"] = target_name
    if trading_calendar is not None:
        params["trading_calendar"] = trading_calendar
    return get_recipe(target_name, **params)


def _log_target_stats(target_block: pd.DataFrame, target_name: str) -> None:
    label_col = f"label.{target_name}"
    valid_col = f"label_valid.{target_name}"
    reason_col = f"label_reason.{target_name}"

    n = len(target_block)
    n_valid = target_block[valid_col].sum()
    n_invalid = n - n_valid

    logger.info(
        "target_block '%s': %d rows — %d valid (%.2f%%), %d invalid (%.2f%%)",
        target_name, n, n_valid, n_valid / n * 100, n_invalid, n_invalid / n * 100,
    )

    if reason_col in target_block.columns:
        reason_counts = target_block.loc[~target_block[valid_col], reason_col].value_counts()
        if len(reason_counts) > 0:
            logger.info("  Invalid reasons:\n%s", reason_counts.to_string())


def build_target_block(
    panel_base: pd.DataFrame,
    target_name: str,
    trading_calendar: pd.DataFrame | None = None,
    **recipe_kwargs,
) -> pd.DataFrame:
    """
    Build a target_block from an in-memory panel_base.

    Parameters
    ----------
    panel_base : DataFrame with index=[date, sid].
    target_name : Registered target name (e.g. "return_5d").
    trading_calendar : Optional trading calendar for recipes that need it.
    **recipe_kwargs : Extra kwargs passed to the recipe constructor
        (overrides spec.params).

    Returns
    -------
    target_block : DataFrame with index=[date, sid] and columns:
        label.<target_name>, label_valid.<target_name>, label_reason.<target_name>
    """
    recipe = _make_recipe(target_name, trading_calendar, **recipe_kwargs)
    recipe.validate_inputs(panel_base)

    logger.info("Building target_block for '%s' (recipe=%s)", target_name, type(recipe).__name__)

    target_block = recipe.compute(panel_base)
    validate_target_block(target_block, target_name)
    _log_target_stats(target_block, target_name)

    return target_block


def build_target_block_from_path(
    panel_base_path: str | Path,
    target_name: str,
    trading_calendar: pd.DataFrame | None = None,
    **recipe_kwargs,
) -> pd.DataFrame:
    """
    Build a target_block by loading only the columns the recipe needs.

    This avoids loading the full panel_base (~14 GB) into memory.
    The recipe's ``required_columns()`` determines which columns to read;
    only those plus [date, sid] are loaded from the parquet file.

    Parameters
    ----------
    panel_base_path : Path to panel_base.parquet.
    target_name : Registered target name.
    trading_calendar : Optional trading calendar.
    **recipe_kwargs : Extra kwargs for the recipe.
    """
    recipe = _make_recipe(target_name, trading_calendar, **recipe_kwargs)

    # Only load what this recipe actually needs
    cols = sorted(set(recipe.required_columns()) | {"date", "sid"})

    logger.info(
        "Loading slim panel_base for '%s': columns=%s",
        target_name, cols,
    )

    table = pq.read_table(str(panel_base_path), columns=cols)
    panel_base = table.to_pandas()
    del table

    # date/sid may already be restored as index by pandas metadata
    if isinstance(panel_base.index, pd.MultiIndex):
        pass  # already [date, sid] index
    elif "date" in panel_base.columns and "sid" in panel_base.columns:
        panel_base = panel_base.set_index(["date", "sid"])

    logger.info(
        "  panel_base (slim): %d rows × %d cols",
        len(panel_base), len(panel_base.columns),
    )

    recipe.validate_inputs(panel_base)

    logger.info("Building target_block for '%s' (recipe=%s)", target_name, type(recipe).__name__)

    target_block = recipe.compute(panel_base)
    validate_target_block(target_block, target_name)
    _log_target_stats(target_block, target_name)

    return target_block
