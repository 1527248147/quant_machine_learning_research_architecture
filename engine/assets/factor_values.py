"""
Build factor_values asset table from normalised factor data.

factor_values schema:
  date, sid (as columns),
  feature.<name>, … (all factor columns)
"""
from __future__ import annotations
import logging

import pandas as pd

from engine.schema.validators import validate_factor_values

logger = logging.getLogger(__name__)


def build_factor_values(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the factor_values asset table from a normalised factor long-table.

    Parameters
    ----------
    factor_df:
        Output of FactorSource.load() with columns [date, sid, feature.*].

    Returns
    -------
    factor_values DataFrame with columns [date, sid, feature.*].
    """
    df = factor_df

    # Ensure feature.* prefix on all non-metadata columns
    meta = {"date", "sid"}
    rename = {}
    for col in df.columns:
        if col not in meta and not col.startswith("feature."):
            rename[col] = f"feature.{col}"
    if rename:
        df = df.rename(columns=rename)

    validate_factor_values(df)

    feat_cols = [c for c in df.columns if c.startswith("feature.")]
    logger.info(
        "factor_values: %d rows, %d unique sids, %d features",
        len(df),
        df["sid"].nunique(),
        len(feat_cols),
    )
    return df
