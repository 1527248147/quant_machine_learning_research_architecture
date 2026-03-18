"""
Parquet I/O utilities.

Supports loading single parquet files and scanning yearly-partitioned
directories (year=2005.parquet, year=2006.parquet, …).
"""
from __future__ import annotations
import gc
import glob
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a single parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    logger.debug("Loaded %s → %d rows × %d cols", path.name, len(df), len(df.columns))
    return df


def load_parquet_daterange(
    path: str | Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load rows from a parquet file where ``date`` is in [start_date, end_date).

    Uses PyArrow row-group statistics for efficient pruning — if the file was
    written one year per row-group (as build_assets does), only the matching
    row-group is read from disk.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    filters = [("date", ">=", start_date), ("date", "<", end_date)]
    table = pq.read_table(str(path), filters=filters, columns=columns)
    df = table.to_pandas()
    logger.debug(
        "Loaded %s [%s, %s) → %d rows",
        path.name, start_date.date(), end_date.date(), len(df),
    )
    return df


def save_parquet(df: pd.DataFrame, path: str | Path, index: bool = True) -> None:
    """Save a DataFrame to parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)
    logger.info("Saved %s (%d rows × %d cols)", path, len(df), len(df.columns))


def _filter_yearly_files(
    directory: Path,
    pattern: str,
    start_year: Optional[int],
    end_year: Optional[int],
) -> List[Tuple[int, str]]:
    """Return sorted list of (year, filepath) within the year range."""
    files = sorted(glob.glob(str(directory / pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {directory}"
        )
    result = []
    for fp in files:
        year_str = Path(fp).stem.split("=")[-1]
        try:
            year = int(year_str)
        except ValueError:
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        result.append((year, fp))
    if not result:
        raise FileNotFoundError(
            f"No yearly files in range [{start_year}, {end_year}] found in {directory}"
        )
    return result


def iter_yearly_parquets(
    directory: str | Path,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    pattern: str = "year=*.parquet",
    columns: Optional[List[str]] = None,
) -> Iterator[Tuple[int, pd.DataFrame]]:
    """
    Yield ``(year, df)`` one year at a time without keeping all years in memory.

    Parameters
    ----------
    columns:
        If provided, only these columns are loaded from each parquet file.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    yearly_files = _filter_yearly_files(directory, pattern, start_year, end_year)
    logger.info("Streaming %d yearly parquet file(s) from %s …", len(yearly_files), directory)

    for year, fp in yearly_files:
        df = pd.read_parquet(fp, columns=columns)
        logger.debug("  %s → %d rows", Path(fp).name, len(df))
        yield year, df


def load_yearly_parquets(
    directory: str | Path,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    pattern: str = "year=*.parquet",
) -> pd.DataFrame:
    """
    Load all yearly parquet files from a directory and concatenate them.

    Files are expected to follow the naming convention ``year=YYYY.parquet``.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    yearly_files = _filter_yearly_files(directory, pattern, start_year, end_year)
    logger.info("Loading %d yearly parquet file(s) from %s …", len(yearly_files), directory)

    frames = []
    for year, fp in yearly_files:
        df = pd.read_parquet(fp)
        frames.append(df)
        logger.debug("  %s → %d rows", Path(fp).name, len(df))

    result = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    logger.info("  Total: %d rows", len(result))
    return result
