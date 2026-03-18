"""
Panel reader utilities for view builders.

Provides high-level functions to read panel_base + target_blocks
via PyArrow column selection, so view builders don't need to handle
parquet I/O details themselves.

Typical usage in a view builder::

    from engine.io.panel_reader import read_panel_index, read_feature_chunks, read_target_labels

    index = read_panel_index(paths)
    for chunk_cols, chunk_vals in read_feature_chunks(paths, chunk_size=50):
        ...
    labels = read_target_labels(paths, "return_1d")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from engine.io.paths import PathManager

logger = logging.getLogger(__name__)


# ======================================================================
# Schema inspection
# ======================================================================

def read_feature_columns(panel_base_path: str | Path) -> List[str]:
    """Read the list of feature.* column names from the parquet schema (no data loaded)."""
    schema = pq.read_schema(str(panel_base_path))
    feat_cols = sorted([f.name for f in schema if f.name.startswith("feature.")])
    logger.debug("Found %d feature columns in schema", len(feat_cols))
    return feat_cols


def read_all_column_names(panel_base_path: str | Path) -> List[str]:
    """Read all column names from the parquet schema (no data loaded)."""
    schema = pq.read_schema(str(panel_base_path))
    return [f.name for f in schema]


# ======================================================================
# Panel index
# ======================================================================

@dataclass
class PanelIndex:
    """
    Lightweight index structure for the panel.

    Provides date/sid axes, (row → t,n) mappings, and optionally
    extra columns read alongside the index (e.g. status columns).

    Attributes
    ----------
    unique_dates : sorted array of unique dates
    unique_sids  : sorted array of unique sids
    T, N         : panel dimensions
    row_t        : int32 array mapping each row to its date index t
    row_n        : int32 array mapping each row to its sid index n
    date_strings : list of date strings (YYYY-MM-DD)
    sid_strings  : list of sid strings
    date_to_t    : dict mapping date value → t index
    sid_to_n     : dict mapping sid value → n index
    extra_columns: dict of extra column arrays read alongside the index
    """
    unique_dates: np.ndarray
    unique_sids: np.ndarray
    T: int
    N: int
    row_t: np.ndarray
    row_n: np.ndarray
    date_strings: List[str]
    sid_strings: List[str]
    date_to_t: Dict = field(default_factory=dict)
    sid_to_n: Dict = field(default_factory=dict)
    extra_columns: Dict[str, np.ndarray] = field(default_factory=dict)
    num_rows: int = 0


def read_panel_index(
    paths: PathManager,
    extra_columns: Optional[List[str]] = None,
) -> PanelIndex:
    """
    Read the panel_base index (date, sid) and optionally extra columns.

    This is lightweight — only reads the index + requested columns,
    not the full 496-column panel_base.

    Parameters
    ----------
    paths : PathManager
    extra_columns : Additional columns to read (e.g. ["status.sample_usable_for_feature"])

    Returns
    -------
    PanelIndex with (row → t,n) mappings and any extra columns.
    """
    cols_to_read = ["date", "sid"]
    if extra_columns:
        cols_to_read.extend(extra_columns)

    df = pq.read_table(
        str(paths.panel_base_path), columns=cols_to_read,
    ).to_pandas()

    # date/sid may come back as pandas index (not columns)
    if "date" not in df.columns:
        df = df.reset_index()

    dates_raw = df["date"]
    sids_raw = df["sid"]

    unique_dates = np.sort(dates_raw.unique())
    unique_sids = np.sort(sids_raw.unique())
    T = len(unique_dates)
    N = len(unique_sids)

    date_to_t = {d: t for t, d in enumerate(unique_dates)}
    sid_to_n = {s: n for n, s in enumerate(unique_sids)}

    row_t = np.array([date_to_t[d] for d in dates_raw], dtype=np.int32)
    row_n = np.array([sid_to_n[s] for s in sids_raw], dtype=np.int32)

    date_strings = [str(d)[:10] for d in unique_dates]
    sid_strings = [str(s) for s in unique_sids]

    extra = {}
    if extra_columns:
        for col in extra_columns:
            if col in df.columns:
                extra[col] = df[col].values

    num_rows = len(df)
    del df, dates_raw, sids_raw

    logger.info("PanelIndex: T=%d dates, N=%d instruments, %d rows", T, N, num_rows)

    return PanelIndex(
        unique_dates=unique_dates,
        unique_sids=unique_sids,
        T=T, N=N,
        row_t=row_t, row_n=row_n,
        date_strings=date_strings,
        sid_strings=sid_strings,
        date_to_t=date_to_t,
        sid_to_n=sid_to_n,
        extra_columns=extra,
        num_rows=num_rows,
    )


# ======================================================================
# Feature chunk reader
# ======================================================================

def read_feature_chunks(
    paths: PathManager,
    chunk_size: int = 50,
    feat_columns: Optional[List[str]] = None,
) -> Iterator[Tuple[int, int, List[str], np.ndarray]]:
    """
    Yield feature columns in chunks from panel_base.

    Each yield provides (chunk_start, chunk_end, column_names, values_f32).
    values_f32 has shape [num_rows, chunk_size].

    Parameters
    ----------
    paths : PathManager
    chunk_size : Number of feature columns per chunk
    feat_columns : Explicit list of feature columns. If None, auto-detected from schema.
    """
    if feat_columns is None:
        feat_columns = read_feature_columns(paths.panel_base_path)

    F = len(feat_columns)
    panel_path = str(paths.panel_base_path)

    for chunk_start in range(0, F, chunk_size):
        chunk_end = min(chunk_start + chunk_size, F)
        chunk_cols = feat_columns[chunk_start:chunk_end]

        logger.info("  Reading feature chunk [%d:%d] / %d ...", chunk_start, chunk_end, F)
        chunk_df = pq.read_table(panel_path, columns=chunk_cols).to_pandas()
        chunk_vals = chunk_df.values.astype(np.float32)
        del chunk_df

        yield chunk_start, chunk_end, chunk_cols, chunk_vals
        del chunk_vals


# ======================================================================
# Target label reader
# ======================================================================

@dataclass
class TargetLabels:
    """Labels and validity mask for a single target."""
    target_name: str
    label_values: np.ndarray      # float32
    valid_mask: np.ndarray         # bool
    found: bool = True


def read_target_labels(
    paths: PathManager,
    target_name: str,
) -> TargetLabels:
    """
    Read label + valid columns from a target_block parquet.

    Parameters
    ----------
    paths : PathManager
    target_name : e.g. "return_1d" or "momentum_cls"

    Returns
    -------
    TargetLabels with label_values and valid_mask arrays.
    If the target_block file doesn't exist, returns TargetLabels with found=False.
    """
    tb_path = paths.target_block_path(target_name)
    label_col = f"label.{target_name}"
    valid_col = f"label_valid.{target_name}"

    if not tb_path.exists():
        logger.warning("Target block not found: %s", tb_path)
        return TargetLabels(
            target_name=target_name,
            label_values=np.array([], dtype=np.float32),
            valid_mask=np.array([], dtype=bool),
            found=False,
        )

    logger.info("Reading target_block: %s", tb_path)
    tb_df = pq.read_table(str(tb_path), columns=[label_col, valid_col]).to_pandas()

    label_values = tb_df[label_col].values.astype(np.float32)
    valid_mask = tb_df[valid_col].values.astype(bool)
    del tb_df

    logger.info("  %s: %d valid / %d total (%.1f%%)",
                target_name, valid_mask.sum(), len(valid_mask),
                100.0 * valid_mask.sum() / len(valid_mask))

    return TargetLabels(
        target_name=target_name,
        label_values=label_values,
        valid_mask=valid_mask,
        found=True,
    )


# ======================================================================
# Convenience: read specific columns from panel_base
# ======================================================================

def read_panel_columns(
    paths: PathManager,
    columns: List[str],
) -> np.ndarray:
    """
    Read specific columns from panel_base as a numpy array.

    Parameters
    ----------
    paths : PathManager
    columns : Column names to read

    Returns
    -------
    np.ndarray of shape [num_rows, len(columns)], dtype float32
    """
    df = pq.read_table(str(paths.panel_base_path), columns=columns).to_pandas()
    vals = df.values.astype(np.float32)
    del df
    return vals
