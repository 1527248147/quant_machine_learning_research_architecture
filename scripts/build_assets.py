"""
scripts/build_assets.py

Step 1 of the pipeline: build all asset tables from raw data.

The only two inputs are:
  - OHLCV yearly parquet directory
  - Factor yearly parquet directory

The trading calendar is derived from the unique dates in the OHLCV data.

Usage:
    python scripts/build_assets.py --config configs/default.yaml

Produces:
    data/processed/assets/instrument_master.parquet
    data/processed/assets/trading_calendar.parquet
    data/processed/assets/daily_bars.parquet
    data/processed/assets/factor_values.parquet
    data/processed/assets/status_intervals.parquet
"""
from __future__ import annotations
import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import yaml
except ImportError:
    yaml = None

# Allow running from framework root
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.sources.ohlcv import OHLCVSource
from engine.sources.factors import FactorSource
from engine.assets.instrument_master import build_instrument_master
from engine.assets.trading_calendar import build_trading_calendar_from_ohlcv
from engine.assets.daily_bars import build_daily_bars
from engine.assets.factor_values import build_factor_values
from engine.assets.status_intervals import build_status_intervals
from engine.io.paths import PathManager
from engine.io.parquet_io import save_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_assets")


def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required for --config support: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build asset tables from raw data.")
    p.add_argument("--config", default=None, help="Path to YAML config file.")
    p.add_argument("--ohlcv-dir", default=None)
    p.add_argument("--factor-dir", default=None)
    p.add_argument("--processed-dir", default=None)
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    args = p.parse_args()

    # Load config and fill in unset args
    cfg: dict = {}
    if args.config:
        cfg = _load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    panel_cfg = cfg.get("panel", {})

    if args.ohlcv_dir is None:
        args.ohlcv_dir = data_cfg.get("raw_ohlcv_dir", r"C:\AI_STOCK\dataset\rq_ohlcv_yearly_parquet2005-2025")
    if args.factor_dir is None:
        args.factor_dir = data_cfg.get("raw_factor_dir", r"C:\AI_STOCK\dataset\alpha158_plus_fund_yearly_parquet")
    if args.processed_dir is None:
        args.processed_dir = data_cfg.get("processed_dir", "data/processed")
    if args.start_year is None:
        args.start_year = panel_cfg.get("start_year", None)
    if args.end_year is None:
        args.end_year = panel_cfg.get("end_year", None)
    if args.start_date is None:
        args.start_date = panel_cfg.get("start_date", None)
    if args.end_date is None:
        args.end_date = panel_cfg.get("end_date", None)

    return args


def _open_writer(
    writer: Optional[pq.ParquetWriter],
    table: pa.Table,
    path: Path,
) -> pq.ParquetWriter:
    """Open a ParquetWriter on first call, reuse on subsequent calls.
    Casts table to the established schema to handle empty-chunk type drift
    (e.g. an empty sid column inferred as null instead of string).
    """
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(path), table.schema)
    else:
        if not table.schema.equals(writer.schema):
            table = table.cast(writer.schema)
    writer.write_table(table)
    return writer


def main() -> None:
    args = parse_args()
    t0 = time.time()

    paths = PathManager(processed_dir=args.processed_dir)
    paths.makedirs()

    ohlcv_src = OHLCVSource(
        raw_dir=args.ohlcv_dir,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    factor_src = FactorSource(
        raw_dir=args.factor_dir,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    # ------------------------------------------------------------------
    # Phase A: lightweight OHLCV scan → instrument_master + trading_calendar
    # Peak memory: only [date, sid] columns across all years (~200 MB)
    # ------------------------------------------------------------------
    logger.info("=== Phase A: instrument_master + trading_calendar (lightweight scan) ===")
    ohlcv_light = ohlcv_src.load_lightweight()

    instrument_master = build_instrument_master(ohlcv_light)
    save_parquet(instrument_master, paths.instrument_master_path, index=False)
    del instrument_master

    trading_calendar = build_trading_calendar_from_ohlcv(
        ohlcv_light, start_date=args.start_date, end_date=args.end_date
    )
    save_parquet(trading_calendar, paths.trading_calendar_path, index=False)
    del trading_calendar, ohlcv_light
    gc.collect()

    # ------------------------------------------------------------------
    # Phase B: stream OHLCV year by year → daily_bars + status_intervals
    # Peak memory: ~1 year of OHLCV (~100 MB)
    # ------------------------------------------------------------------
    logger.info("=== Phase B: daily_bars + status_intervals (year-by-year streaming) ===")
    bars_writer: Optional[pq.ParquetWriter] = None
    susp_writer: Optional[pq.ParquetWriter] = None

    for year, ohlcv_year in ohlcv_src.iter_years():
        logger.info("  OHLCV year=%d: %d rows", year, len(ohlcv_year))

        bars_year = build_daily_bars(ohlcv_year)
        susp_year = build_status_intervals(bars_year)
        del ohlcv_year

        bars_writer = _open_writer(bars_writer, pa.Table.from_pandas(bars_year, preserve_index=False), paths.daily_bars_path)
        susp_writer = _open_writer(susp_writer, pa.Table.from_pandas(susp_year, preserve_index=False), paths.status_intervals_path)
        del bars_year, susp_year
        gc.collect()

    if bars_writer:
        bars_writer.close()
        logger.info("Saved %s", paths.daily_bars_path)
    if susp_writer:
        susp_writer.close()
        logger.info("Saved %s", paths.status_intervals_path)

    # ------------------------------------------------------------------
    # Phase C: stream factors year by year → factor_values
    # Peak memory: ~1 year of factors (~450 MB)
    # ------------------------------------------------------------------
    logger.info("=== Phase C: factor_values (year-by-year streaming) ===")
    fact_writer: Optional[pq.ParquetWriter] = None

    for year, factor_year in factor_src.iter_years():
        logger.info("  factors year=%d: %d rows", year, len(factor_year))

        values_year = build_factor_values(factor_year)
        del factor_year

        fact_writer = _open_writer(fact_writer, pa.Table.from_pandas(values_year, preserve_index=False), paths.factor_values_path)
        del values_year
        gc.collect()

    if fact_writer:
        fact_writer.close()
        logger.info("Saved %s", paths.factor_values_path)

    elapsed = time.time() - t0
    logger.info("=== build_assets complete in %.1fs ===", elapsed)
    logger.info("Outputs written to: %s", paths.assets_dir)


if __name__ == "__main__":
    main()
