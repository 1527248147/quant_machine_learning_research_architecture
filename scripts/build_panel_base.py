"""
scripts/build_panel_base.py

Step 2 of the pipeline: build active_session_index and panel_base.

Reads the asset tables produced by build_assets.py and assembles the
unified panel_base with all status.* columns.

**Memory-efficient**: processes one year at a time via PyArrow row-group
filtering + ParquetWriter streaming.  Peak memory ≈ 1-2 GB per year
instead of loading all factor_values at once (~28 GB).

Usage:
    python scripts/build_panel_base.py --config configs/default.yaml
"""
from __future__ import annotations
import argparse
import gc
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import yaml
except ImportError:
    yaml = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.io.paths import PathManager
from engine.io.parquet_io import load_parquet, load_parquet_daterange, save_parquet
from engine.panel.index_builder import build_active_session_index
from engine.panel.status_builder import build_status_columns
from engine.panel.build_panel_base import assemble_panel_chunk
from engine.core.constants import SampleState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_panel_base")


def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required for --config support: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build active_session_index and panel_base.")
    p.add_argument("--config", default=None, help="Path to YAML config file.")
    p.add_argument("--processed-dir", default=None)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--report-only", action="store_true",
                   help="Skip building, only regenerate guide + column reference from existing parquet files.")
    args = p.parse_args()

    cfg: dict = {}
    if args.config:
        cfg = _load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    panel_cfg = cfg.get("panel", {})

    if args.processed_dir is None:
        args.processed_dir = data_cfg.get("processed_dir", "data/processed")
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
    """Open a ParquetWriter on first call, reuse on subsequent calls."""
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(path), table.schema)
    else:
        if not table.schema.equals(writer.schema):
            table = table.cast(writer.schema)
    writer.write_table(table)
    return writer


# ======================================================================
# Incremental stats accumulator (avoids loading full panel for reports)
# ======================================================================
class PanelStats:
    """Accumulates lightweight stats across yearly chunks."""

    def __init__(self):
        self.total_rows = 0
        self.sample_state_counts: Counter = Counter()
        self.bool_true_counts: defaultdict = defaultdict(int)
        self.fmr_min = 1.0
        self.fmr_max = 0.0
        self.fmr_has_variety = False
        self.market_state_values: set = set()
        self.factor_state_values: set = set()

    def update(self, panel_chunk: pd.DataFrame) -> None:
        n = len(panel_chunk)
        self.total_rows += n

        if "status.sample_state" in panel_chunk.columns:
            for state, cnt in panel_chunk["status.sample_state"].value_counts().items():
                self.sample_state_counts[state] += cnt

        bool_cols = [
            "status.is_listed", "status.is_suspended",
            "status.has_market_record", "status.has_factor_record",
            "status.bar_missing", "status.factor_row_missing",
            "status.feature_all_missing", "status.sample_usable_for_feature",
        ]
        for col in bool_cols:
            if col in panel_chunk.columns:
                self.bool_true_counts[col] += int(panel_chunk[col].sum())

        if "status.factor_missing_ratio" in panel_chunk.columns:
            fmr = panel_chunk["status.factor_missing_ratio"]
            self.fmr_min = min(self.fmr_min, float(fmr.min()))
            self.fmr_max = max(self.fmr_max, float(fmr.max()))
            if fmr.nunique() > 1:
                self.fmr_has_variety = True

        if "status.market_state" in panel_chunk.columns:
            self.market_state_values.update(panel_chunk["status.market_state"].unique())
        if "status.factor_state" in panel_chunk.columns:
            self.factor_state_values.update(panel_chunk["status.factor_state"].unique())

    def print_quality_report(self) -> None:
        logger.info("=== Panel Base Quality Report ===")
        n = self.total_rows
        if n == 0:
            logger.warning("panel_base is empty (0 rows)")
            return

        for state in sorted(self.sample_state_counts, key=lambda s: -self.sample_state_counts[s]):
            cnt = self.sample_state_counts[state]
            pct = cnt / n * 100
            logger.info("  sample_state=%-30s %10d  (%6.2f%%)", state, cnt, pct)

        for col in sorted(self.bool_true_counts):
            rate = self.bool_true_counts[col] / n * 100
            logger.info("  %-45s %6.2f%%", col, rate)

    def validate_sanity(self) -> list[str]:
        """Run sanity checks on accumulated stats. Returns list of issues."""
        issues: list[str] = []
        n = self.total_rows
        if n == 0:
            issues.append("panel_base is empty (0 rows)")
            return issues

        # Boolean columns: should have both True and False
        for col in self.bool_true_counts:
            true_count = self.bool_true_counts[col]
            if col == "status.is_listed":
                continue  # always True by construction
            if true_count == 0:
                issues.append(f"SUSPICIOUS: {col} is all False (0/{n})")
            elif true_count == n:
                issues.append(f"SUSPICIOUS: {col} is all True ({n}/{n})")

        # Suspended must exist
        susp_count = self.bool_true_counts.get("status.is_suspended", 0)
        if susp_count == 0:
            issues.append(
                "FAIL: status.is_suspended is all False — "
                "impossible for A-share data (suspended stocks are expected)"
            )
        elif susp_count / n > 0.5:
            issues.append(
                f"SUSPICIOUS: is_suspended rate = {susp_count/n*100:.1f}% — "
                f"more than half the panel is suspended"
            )

        # has_market_record majority True
        mkt_count = self.bool_true_counts.get("status.has_market_record", 0)
        if mkt_count / n < 0.5:
            issues.append(
                f"SUSPICIOUS: has_market_record rate = {mkt_count/n*100:.1f}% — "
                f"less than half the panel has market data"
            )

        # factor_missing_ratio range
        if not self.fmr_has_variety:
            issues.append(
                f"SUSPICIOUS: factor_missing_ratio has no variety "
                f"(min={self.fmr_min:.4f}, max={self.fmr_max:.4f})"
            )

        # sample_state diversity
        states = self.sample_state_counts
        if len(states) < 2:
            issues.append(f"SUSPICIOUS: sample_state has only {len(states)} value(s)")
        if SampleState.SUSPENDED.value not in states:
            issues.append("FAIL: sample_state never contains SUSPENDED")

        # market_state
        from engine.core.constants import MarketState
        if MarketState.OK.value not in self.market_state_values:
            issues.append("FAIL: market_state never contains OK")
        if len(self.market_state_values) < 2:
            issues.append(f"SUSPICIOUS: market_state has only {self.market_state_values}")

        # factor_state — with many features, PARTIAL_MISSING is expected;
        # only warn if factor_state has a single value (no variety at all)
        if len(self.factor_state_values) < 2:
            issues.append(
                f"SUSPICIOUS: factor_state has only {self.factor_state_values}"
            )

        # sample_usable_for_feature
        usable_count = self.bool_true_counts.get("status.sample_usable_for_feature", 0)
        if usable_count == 0:
            issues.append("FAIL: sample_usable_for_feature is all False — nothing can be trained")
        elif usable_count == n:
            issues.append("SUSPICIOUS: sample_usable_for_feature is all True (no unusable samples?)")

        if issues:
            logger.warning(
                "Sanity validation: %d issue(s):\n  %s",
                len(issues), "\n  ".join(issues),
            )
        else:
            logger.info("All sanity checks passed (%d rows).", n)

        return issues


def _report_only(paths: PathManager) -> None:
    """Regenerate guide + column reference from existing panel_base parquet."""
    from engine.panel.report_generator import generate_guide, generate_column_reference

    panel_path = paths.panel_base_path
    if not panel_path.exists():
        logger.error("panel_base.parquet not found at %s — run full build first.", panel_path)
        return

    # Compute stats from existing parquet (read only status columns, not full data)
    logger.info("=== Reading status columns from existing panel_base ===")
    status_cols_to_read = [
        "status.sample_state", "status.is_listed", "status.is_suspended",
        "status.has_market_record", "status.has_factor_record",
        "status.bar_missing", "status.factor_row_missing",
        "status.feature_all_missing", "status.sample_usable_for_feature",
        "status.factor_missing_ratio",
    ]
    panel_status = pd.read_parquet(panel_path, columns=status_cols_to_read)
    total_rows = len(panel_status)

    sample_state_counts = {}
    if "status.sample_state" in panel_status.columns:
        for state, cnt in panel_status["status.sample_state"].value_counts().items():
            sample_state_counts[state] = int(cnt)

    bool_cols = [c for c in status_cols_to_read if c != "status.sample_state" and c != "status.factor_missing_ratio"]
    bool_true_counts = {}
    for col in bool_cols:
        if col in panel_status.columns:
            bool_true_counts[col] = int(panel_status[col].sum())

    del panel_status

    stats_dict = {
        "total_rows": total_rows,
        "sample_state_counts": sample_state_counts,
        "bool_true_counts": bool_true_counts,
        "sanity_issues": [],
    }

    generate_guide(paths.panel_dir, paths.assets_dir, stats_dict)
    generate_column_reference(paths.panel_dir, paths.assets_dir)
    logger.info("Done. Reports saved to %s", paths.panel_dir)


def main() -> None:
    args = parse_args()
    t0 = time.time()

    paths = PathManager(processed_dir=args.processed_dir)
    paths.ensure_dir(paths.panel_dir)

    # ------------------------------------------------------------------
    # Report-only mode: skip building, just regenerate docs
    # ------------------------------------------------------------------
    if args.report_only:
        _report_only(paths)
        return

    # ------------------------------------------------------------------
    # 1. Load lightweight asset tables (small — fit easily in memory)
    # ------------------------------------------------------------------
    logger.info("=== Loading lightweight asset tables ===")
    instrument_master = load_parquet(paths.instrument_master_path)
    trading_calendar = load_parquet(paths.trading_calendar_path)

    for df in [instrument_master, trading_calendar]:
        for col in ("date", "list_date", "delist_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # Apply date range filter
    if args.start_date:
        trading_calendar = trading_calendar[
            trading_calendar["date"] >= pd.Timestamp(args.start_date)
        ]
    if args.end_date:
        trading_calendar = trading_calendar[
            trading_calendar["date"] <= pd.Timestamp(args.end_date)
        ]

    # ------------------------------------------------------------------
    # 2. Build active_session_index (full — ~27M rows × 2 cols ≈ 660 MB)
    # ------------------------------------------------------------------
    logger.info("=== Building active_session_index ===")
    active_idx = build_active_session_index(instrument_master, trading_calendar)
    save_parquet(active_idx, paths.active_session_index_path, index=False)

    # ------------------------------------------------------------------
    # 3. Year-by-year panel assembly
    # ------------------------------------------------------------------
    years = sorted(active_idx["date"].dt.year.unique())
    logger.info(
        "=== Building panel_base year-by-year (%d years: %d–%d) ===",
        len(years), years[0], years[-1],
    )

    panel_writer: Optional[pq.ParquetWriter] = None
    stats = PanelStats()

    for year in years:
        t_year = time.time()

        # 3a. Filter active_session_index to this year
        mask = active_idx["date"].dt.year == year
        idx_year = active_idx[mask].copy()

        # 3b. Load daily_bars for this year (pyarrow filtered read)
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year + 1}-01-01")
        bars_year = load_parquet_daterange(
            paths.daily_bars_path, year_start, year_end,
        )

        # 3c. Load factor_values for this year (pyarrow filtered read)
        factor_year = load_parquet_daterange(
            paths.factor_values_path, year_start, year_end,
        )

        # 3d. Build status columns
        status_year = build_status_columns(
            active_session_index=idx_year,
            daily_bars=bars_year,
            factor_values=factor_year,
        )

        # 3e. Assemble panel chunk
        panel_year = assemble_panel_chunk(
            active_session_index=idx_year,
            daily_bars=bars_year,
            factor_values=factor_year,
            status_df=status_year,
            instrument_master=instrument_master,
        )

        # 3f. Accumulate stats
        stats.update(panel_year)

        # 3g. Write chunk via ParquetWriter
        table = pa.Table.from_pandas(panel_year, preserve_index=True)
        panel_writer = _open_writer(panel_writer, table, paths.panel_base_path)

        elapsed_year = time.time() - t_year
        logger.info(
            "  year=%d: %d rows, %.1fs",
            year, len(panel_year), elapsed_year,
        )

        del idx_year, bars_year, factor_year, status_year, panel_year, table
        gc.collect()

    if panel_writer:
        panel_writer.close()

    # ------------------------------------------------------------------
    # 4. Quality report + sanity validation (from accumulated stats)
    # ------------------------------------------------------------------
    stats.print_quality_report()
    issues = stats.validate_sanity()
    if issues:
        logger.warning("Sanity validation found %d issue(s).", len(issues))
    else:
        logger.info("All sanity checks passed.")

    # ------------------------------------------------------------------
    # 5. Auto-generate usage guide + column reference
    # ------------------------------------------------------------------
    logger.info("=== Generating usage guide ===")
    from engine.panel.report_generator import generate_guide, generate_column_reference

    stats_dict = {
        "total_rows": stats.total_rows,
        "sample_state_counts": dict(stats.sample_state_counts),
        "bool_true_counts": dict(stats.bool_true_counts),
        "sanity_issues": issues,
    }
    generate_guide(paths.panel_dir, paths.assets_dir, stats_dict)
    generate_column_reference(paths.panel_dir, paths.assets_dir)

    elapsed = time.time() - t0
    logger.info("=== build_panel_base complete in %.1fs ===", elapsed)
    logger.info("panel_base saved to: %s (%d total rows)", paths.panel_base_path, stats.total_rows)


if __name__ == "__main__":
    main()
