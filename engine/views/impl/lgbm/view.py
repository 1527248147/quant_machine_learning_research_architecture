"""
LightGBM View Builder: builds a 2D tabular dataset from panel_base + target_blocks.

Reads source parquet files directly via PyArrow column selection — never loads
the full panel_base into memory at once.

Output structure (numpy arrays saved to disk):
    X_f32.npy         : [num_rows, F]  float32  — feature matrix
    y_f32.npy         : [num_rows]     float32  — label values
    y_valid_u8.npy    : [num_rows]     uint8    — label validity mask
    dates.npy         : [num_rows]     object   — date string per row
    sids.npy          : [num_rows]     object   — sid per row
    row_usable_u8.npy : [num_rows]     uint8    — sample_usable_for_feature
    meta.json         : metadata (F, dates, instruments, feat_cols, ...)

The view stores raw panel data. Grouping, relevance binning, and
train/valid/test splitting happen at training time in the trainer.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq

from engine.io.paths import PathManager
from engine.io.panel_reader import (
    read_feature_columns,
    read_panel_index,
    read_feature_chunks,
    read_target_labels,
)
from engine.views.base import BaseViewBuilder

logger = logging.getLogger(__name__)


def build_tabular_view(
    paths: PathManager,
    out_dir: Path,
    label_name: str = "return_c1c2",
    config: dict | None = None,
) -> Dict:
    """
    Build 2D tabular view from panel_base + target_block.

    Reads data in chunks to keep memory manageable. Stores the full
    panel as row-oriented numpy arrays, letting the trainer handle
    date-based splitting and grouping at training time.

    Returns the meta dict (also saved as meta.json).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # --- 1. Discover feature columns ----------------------------------------
    all_feat_columns = read_feature_columns(paths.panel_base_path)

    # Apply feature filtering from config
    view_cfg = (config or {}).get("model", {}).get("view", {})
    build_include = view_cfg.get("include_pattern", None)
    build_exclude = view_cfg.get("exclude_pattern", None)
    fund_features_whitelist = view_cfg.get("fund_features", None)

    feat_columns = list(all_feat_columns)

    # Step 1: regex-based include/exclude
    if build_include:
        pat = re.compile(build_include)
        feat_columns = [c for c in feat_columns if pat.search(c)]
    if build_exclude:
        pat = re.compile(build_exclude)
        feat_columns = [c for c in feat_columns if not pat.search(c)]

    # Step 2: fund feature whitelist — keep all non-fund features,
    # but only keep fund features that are in the whitelist.
    # Fund features are identified by containing "fund__" in the column name.
    if fund_features_whitelist is not None:
        # Build the set of allowed fund column names (with feature. prefix)
        allowed_fund = set()
        for f in fund_features_whitelist:
            if f.startswith("feature."):
                allowed_fund.add(f)
            else:
                allowed_fund.add(f"feature.{f}")

        filtered = []
        n_fund_kept = 0
        n_fund_dropped = 0
        for c in feat_columns:
            if "fund__" in c:
                if c in allowed_fund:
                    filtered.append(c)
                    n_fund_kept += 1
                else:
                    n_fund_dropped += 1
            else:
                filtered.append(c)

        feat_columns = filtered
        logger.info(
            "Fund whitelist filter: kept %d fund features, dropped %d",
            n_fund_kept, n_fund_dropped,
        )

    if build_include or build_exclude or fund_features_whitelist is not None:
        logger.info(
            "Build-time feature filter: %d / %d features",
            len(feat_columns), len(all_feat_columns),
        )

    F = len(feat_columns)
    logger.info("Features: %d", F)

    # --- 2. Read panel index + status ----------------------------------------
    usable_col = "status.sample_usable_for_feature"
    pidx = read_panel_index(paths, extra_columns=[usable_col])
    num_rows = pidx.num_rows
    logger.info("Panel: %d rows, T=%d dates, N=%d instruments", num_rows, pidx.T, pidx.N)

    # Save index arrays
    dates_arr = np.array([pidx.date_strings[t] for t in pidx.row_t], dtype=object)
    sids_arr = np.array([pidx.sid_strings[n] for n in pidx.row_n], dtype=object)
    row_usable = pidx.extra_columns[usable_col].astype(np.uint8)

    np.save(str(out_dir / "dates.npy"), dates_arr)
    np.save(str(out_dir / "sids.npy"), sids_arr)
    np.save(str(out_dir / "row_usable_u8.npy"), row_usable)

    # --- 3. Build feature matrix in chunks -----------------------------------
    logger.info("Building feature matrix [%d, %d] ...", num_rows, F)
    X = np.zeros((num_rows, F), dtype=np.float32)

    for chunk_start, chunk_end, chunk_cols, chunk_vals in read_feature_chunks(
        paths, chunk_size=50, feat_columns=feat_columns,
    ):
        X[:, chunk_start:chunk_end] = chunk_vals
        del chunk_vals

    np.save(str(out_dir / "X_f32.npy"), X)
    del X
    logger.info("Feature matrix saved.")

    # --- 4. Read label -------------------------------------------------------
    labels = read_target_labels(paths, label_name)
    if labels.found:
        np.save(str(out_dir / "y_f32.npy"), labels.label_values)
        np.save(str(out_dir / "y_valid_u8.npy"), labels.valid_mask.astype(np.uint8))
    else:
        logger.warning("Label '%s' not found — saving empty arrays", label_name)
        np.save(str(out_dir / "y_f32.npy"), np.zeros(num_rows, dtype=np.float32))
        np.save(str(out_dir / "y_valid_u8.npy"), np.zeros(num_rows, dtype=np.uint8))

    elapsed = time.time() - t0
    logger.info("Tabular view built in %.1f s", elapsed)

    # --- 5. Save meta --------------------------------------------------------
    meta = {
        "F": F,
        "num_rows": num_rows,
        "T": pidx.T,
        "N": pidx.N,
        "dates": pidx.date_strings,
        "instruments": pidx.sid_strings,
        "feat_cols": feat_columns,
        "label_name": label_name,
        "build_time_s": round(elapsed, 1),
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Meta saved → %s", meta_path)

    return meta


class LGBMTabularDataset:
    """
    In-memory tabular dataset for LightGBM.

    Holds feature matrix X, labels y, validity mask, dates, and sids
    for a specific date range. Provides methods to get LightGBM-ready
    arrays with grouping information for LambdaRank.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_valid: np.ndarray,
        dates: np.ndarray,
        sids: np.ndarray,
        row_usable: np.ndarray,
        feat_cols: List[str],
    ):
        self.X = X
        self.y = y
        self.y_valid = y_valid
        self.dates = dates
        self.sids = sids
        self.row_usable = row_usable
        self.feat_cols = feat_cols

    def get_usable_data(
        self,
        min_group_size: int = 30,
        missing_drop_thresh: float = 0.98,
        exclude_regex: Optional[str] = None,
    ):
        """
        Filter to usable rows (valid label + usable sample) and compute
        feature column mask, returning (X, y_raw, dates, sids, feat_cols_used).

        Does NOT apply relevance binning — that's the trainer's job.
        """
        # Row mask: label valid AND sample usable
        mask = (self.y_valid > 0) & (self.row_usable > 0)
        X = self.X[mask]
        y = self.y[mask]
        dates = self.dates[mask]
        sids = self.sids[mask]

        # Feature column filtering
        feat_cols = list(self.feat_cols)
        col_mask = np.ones(len(feat_cols), dtype=bool)

        if exclude_regex:
            pat = re.compile(exclude_regex)
            col_mask = np.array([not pat.search(c) for c in feat_cols])

        # Drop zero-variance and high-missing features
        X_subset = X[:, col_mask]
        feat_subset = [c for c, m in zip(feat_cols, col_mask) if m]

        var = np.nanvar(X_subset, axis=0)
        keep_var = (var > 0) & np.isfinite(var)

        miss_rate = np.isnan(X_subset).mean(axis=0)
        keep_miss = miss_rate < missing_drop_thresh

        keep = keep_var & keep_miss
        X_final = X_subset[:, keep]
        feat_final = [c for c, k in zip(feat_subset, keep) if k]

        logger.info(
            "Usable data: %d rows, %d features (from %d total rows, %d features)",
            len(X_final), len(feat_final), len(self.X), len(self.feat_cols),
        )

        return X_final, y, dates, sids, feat_final


class LGBMViewBuilder(BaseViewBuilder):
    """View builder for LightGBM LambdaRank model."""

    name = "lgbm_rank"

    def required_columns(self) -> List[str]:
        return []  # dynamic — validated at build time

    def build(
        self,
        paths: PathManager,
        config: dict,
    ) -> Path:
        """Build tabular view from panel_base + target_block."""
        label_roles = config.get("model", {}).get("label_roles", {})
        label_name = label_roles.get("regression", "return_c1c2")

        # For ANY_SINGLE contract, if no label_roles, use first target
        if not label_name:
            targets = config.get("targets", [])
            if targets:
                label_name = targets[0]["name"]

        out_dir = paths.memmap_dir(self.name)
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            logger.info("Tabular view already exists at %s — skipping build", out_dir)
            return out_dir

        logger.info("Building tabular view → %s", out_dir)
        build_tabular_view(
            paths=paths,
            out_dir=out_dir,
            label_name=label_name,
            config=config,
        )
        return out_dir

    def get_dataset(
        self,
        view_dir: Path,
        day_start: int,
        day_end: int,
        config: dict,
    ) -> LGBMTabularDataset:
        """
        Load the view and filter to the given day range.

        Parameters
        ----------
        view_dir : Directory containing the built artifacts.
        day_start, day_end : Inclusive day index range.
        config : Full experiment config dict.
        """
        view_dir = Path(view_dir)

        # Load meta
        with open(view_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        all_dates = meta["dates"]
        target_dates = set(all_dates[day_start:day_end + 1])

        # Load arrays
        X = np.load(str(view_dir / "X_f32.npy"), mmap_mode="r")
        y = np.load(str(view_dir / "y_f32.npy"), mmap_mode="r")
        y_valid = np.load(str(view_dir / "y_valid_u8.npy"), mmap_mode="r")
        dates = np.load(str(view_dir / "dates.npy"), allow_pickle=True)
        sids = np.load(str(view_dir / "sids.npy"), allow_pickle=True)
        row_usable = np.load(str(view_dir / "row_usable_u8.npy"), mmap_mode="r")

        # Filter rows by date range
        date_mask = np.array([d in target_dates for d in dates])
        idx = np.where(date_mask)[0]

        logger.info(
            "LGBMTabularDataset: day_range=[%d,%d] → %d rows (dates: %s ~ %s)",
            day_start, day_end, len(idx),
            all_dates[day_start], all_dates[day_end],
        )

        return LGBMTabularDataset(
            X=np.array(X[idx]),  # copy from mmap to RAM
            y=np.array(y[idx]),
            y_valid=np.array(y_valid[idx]),
            dates=dates[idx],
            sids=sids[idx],
            row_usable=np.array(row_usable[idx]),
            feat_cols=meta["feat_cols"],
        )

    def build_dataloader(self, dataset: Any, config: dict, shuffle: bool = True) -> Any:
        """No-op for LightGBM — dataset is used directly."""
        return dataset
