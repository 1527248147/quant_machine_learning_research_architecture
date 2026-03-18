"""
LSTM View Builder: builds a 3D memmap panel from panel_base + target_blocks.

Reads source parquet files directly via PyArrow column selection — never loads
the full panel_base (~14 GB) into memory.  Only the feature columns, one
status column, and label columns from target_blocks are read.

Output memmap structure (mirrors the LSTM-Momentum reference project):
    X_f16.mmap      : [T, N, D]  float16  — feature panel
    y_ret_f32.mmap  : [T, N]    float32  — regression label
    y_mom_i8.mmap   : [T, N]    int8     — classification label
    ret_mask_u8.mmap: [T, N]    uint8    — regression label valid mask
    mom_mask_u8.mmap: [T, N]    uint8    — classification label valid mask
    both_mask_u8.mmap:[T, N]    uint8    — both labels valid mask
    meta.json       : metadata (T, N, D, dates, instruments, feat_cols, ...)

Feature layout per time-step (D = 2F + 1):
    [features(F), isna_flags(F), row_present(1)]
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore

from engine.io.paths import PathManager
from engine.io.panel_reader import (
    read_feature_columns,
    read_panel_index,
    read_feature_chunks,
    read_target_labels,
)
from engine.views.base import BaseViewBuilder

logger = logging.getLogger(__name__)


# ======================================================================
# Memmap builder
# ======================================================================

def build_memmap_panel(
    paths: PathManager,
    out_dir: Path,
    regression_label: str = "return_c0c1",
    classification_label: str = "momentum_cls",
    config: dict | None = None,
) -> Dict:
    """
    Build 3D memmap panel from panel_base + target_blocks (no panel_labeled needed).

    Uses panel_reader utilities for all I/O, keeping peak memory ~2 GB
    instead of ~14 GB for the full panel_base.

    Returns the meta dict that is also saved as meta.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # --- 1. Discover feature columns from parquet schema -------------------
    all_feat_columns = read_feature_columns(paths.panel_base_path)

    # Apply build-time feature filtering from model.view config
    view_cfg = (config or {}).get("model", {}).get("view", {})
    build_include = view_cfg.get("include_pattern", None)
    build_exclude = view_cfg.get("exclude_pattern", None)

    if build_include or build_exclude:
        import re
        feat_columns = list(all_feat_columns)
        if build_include:
            pat = re.compile(build_include)
            feat_columns = [c for c in feat_columns if pat.search(c)]
        if build_exclude:
            pat = re.compile(build_exclude)
            feat_columns = [c for c in feat_columns if not pat.search(c)]
        logger.info(
            "Build-time feature filter: %d / %d features (include=%s, exclude=%s)",
            len(feat_columns), len(all_feat_columns), build_include, build_exclude,
        )
    else:
        feat_columns = list(all_feat_columns)

    F = len(feat_columns)
    D = 2 * F + 1  # features + isna_flags + row_present
    logger.info("Features: %d, Total input dim D: %d (2*%d+1)", F, D, F)

    # --- 2. Read panel index + status column (lightweight) -----------------
    usable_col = "status.sample_usable_for_feature"
    pidx = read_panel_index(paths, extra_columns=[usable_col])
    T, N = pidx.T, pidx.N
    row_t, row_n = pidx.row_t, pidx.row_n
    row_present_vals = pidx.extra_columns[usable_col].astype(np.float32)
    logger.info("Panel shape: T=%d dates, N=%d instruments, D=%d", T, N, D)

    # --- 3. Create memmap files --------------------------------------------
    def create_mm(name: str, dtype, shape):
        path = str(out_dir / name)
        return np.memmap(path, dtype=dtype, mode="w+", shape=shape)

    X_mm = create_mm("X_f16.mmap", np.float16, (T, N, D))
    y_ret_mm = create_mm("y_ret_f32.mmap", np.float32, (T, N))
    y_mom_mm = create_mm("y_mom_i8.mmap", np.int8, (T, N))
    ret_mask_mm = create_mm("ret_mask_u8.mmap", np.uint8, (T, N))
    mom_mask_mm = create_mm("mom_mask_u8.mmap", np.uint8, (T, N))
    both_mask_mm = create_mm("both_mask_u8.mmap", np.uint8, (T, N))

    # Initialise labels to invalid defaults
    y_ret_mm[:] = 0.0
    y_mom_mm[:] = -1
    ret_mask_mm[:] = 0
    mom_mask_mm[:] = 0
    both_mask_mm[:] = 0

    # --- 4. Fill features in chunks ----------------------------------------
    logger.info("Filling features into memmap (%d rows, %d feature cols) ...", pidx.num_rows, F)

    for chunk_start, chunk_end, chunk_cols, chunk_vals in read_feature_chunks(
        paths, chunk_size=50, feat_columns=feat_columns,
    ):
        isna = np.isnan(chunk_vals).astype(np.float16)
        feats_clean = np.nan_to_num(chunk_vals, nan=0.0).astype(np.float16)
        del chunk_vals

        for t_idx in range(T):
            mask = row_t == t_idx
            if not mask.any():
                continue
            n_indices = row_n[mask]
            X_mm[t_idx, n_indices, chunk_start:chunk_end] = feats_clean[mask]
            X_mm[t_idx, n_indices, F + chunk_start:F + chunk_end] = isna[mask]

        del feats_clean, isna

    # Fill row_present (last dimension)
    logger.info("  Filling row_present ...")
    for t_idx in range(T):
        mask = row_t == t_idx
        if not mask.any():
            continue
        n_indices = row_n[mask]
        X_mm[t_idx, n_indices, D - 1] = row_present_vals[mask].astype(np.float16)

    X_mm.flush()
    logger.info("Features filled.")

    # --- 5. Fill labels from target_blocks ---------------------------------
    ret_labels = read_target_labels(paths, regression_label)
    has_ret_label = ret_labels.found
    if has_ret_label:
        for t_idx in range(T):
            mask = row_t == t_idx
            if not mask.any():
                continue
            n_indices = row_n[mask]
            y_ret_mm[t_idx, n_indices] = np.nan_to_num(ret_labels.label_values[mask], nan=0.0)
            ret_mask_mm[t_idx, n_indices] = ret_labels.valid_mask[mask].astype(np.uint8)
        del ret_labels

    mom_labels = read_target_labels(paths, classification_label)
    has_mom_label = mom_labels.found
    if has_mom_label:
        for t_idx in range(T):
            mask = row_t == t_idx
            if not mask.any():
                continue
            n_indices = row_n[mask]
            vals = mom_labels.label_values[mask]
            valids = mom_labels.valid_mask[mask]
            y_mom_mm[t_idx, n_indices] = np.where(valids, vals, -1).astype(np.int8)
            mom_mask_mm[t_idx, n_indices] = valids.astype(np.uint8)
        del mom_labels

    # Both mask
    if has_ret_label and has_mom_label:
        for t_idx in range(T):
            both_mask_mm[t_idx] = ret_mask_mm[t_idx] & mom_mask_mm[t_idx]

    y_ret_mm.flush()
    y_mom_mm.flush()
    ret_mask_mm.flush()
    mom_mask_mm.flush()
    both_mask_mm.flush()

    elapsed = time.time() - t0
    logger.info("Memmap panel built in %.1f s", elapsed)

    # --- 6. Save meta -------------------------------------------------------
    isna_cols = [f"{c}__isna" for c in feat_columns]
    x_cols = feat_columns + isna_cols + ["row_present"]

    meta = {
        "T": T,
        "N": N,
        "D": D,
        "F": F,
        "dates": pidx.date_strings,
        "instruments": pidx.sid_strings,
        "feat_cols": feat_columns,
        "isna_cols": isna_cols,
        "X_cols": x_cols,
        "has_row_present": True,
        "regression_label": regression_label,
        "classification_label": classification_label,
        "build_time_s": round(elapsed, 1),
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Meta saved → %s", meta_path)

    # --- 7. Generate reports ------------------------------------------------
    _write_column_reference(out_dir, meta)
    _write_view_report(out_dir, meta, X_mm, y_ret_mm, y_mom_mm,
                       ret_mask_mm, mom_mask_mm, both_mask_mm, elapsed)

    return meta


# ======================================================================
# Report generation
# ======================================================================

def _write_column_reference(out_dir: Path, meta: dict) -> None:
    """Write COLUMN_REFERENCE.md listing all dimensions in the memmap."""
    F = meta["F"]
    D = meta["D"]
    feat_cols = meta["feat_cols"]
    isna_cols = meta["isna_cols"]

    lines = [
        "# View Column Reference (lstm_mtl)",
        "",
        f"> Auto-generated by `build_view.py`. Total dimensions: D = {D} (2×{F} + 1)",
        "",
        "## X tensor layout: `X[t, n, :]` — shape `[T, N, D]`",
        "",
        "| Index range | Count | Description |",
        "|---|---|---|",
        f"| `[0, {F-1}]` | {F} | Feature values (NaN → 0.0) |",
        f"| `[{F}, {2*F-1}]` | {F} | isna flags (1.0 if original was NaN) |",
        f"| `[{D-1}]` | 1 | row_present (`status.sample_usable_for_feature`) |",
        "",
        "## Feature columns (index 0 … {})".format(F - 1),
        "",
        "| Dim | Column |",
        "|---|---|",
    ]
    for i, col in enumerate(feat_cols):
        lines.append(f"| {i} | `{col}` |")

    lines += [
        "",
        "## isna flag columns (index {} … {})".format(F, 2 * F - 1),
        "",
        "| Dim | Flags for |",
        "|---|---|",
    ]
    for i, col in enumerate(isna_cols):
        lines.append(f"| {F + i} | `{col}` |")

    lines += [
        "",
        "## Label memmap files",
        "",
        "| File | dtype | Shape | Description |",
        "|---|---|---|---|",
        f"| `y_ret_f32.mmap` | float32 | `[T, N]` | Regression label (`label.{meta['regression_label']}`) |",
        f"| `y_mom_i8.mmap` | int8 | `[T, N]` | Classification label (`label.{meta['classification_label']}`, 0-4, invalid=-1) |",
        f"| `ret_mask_u8.mmap` | uint8 | `[T, N]` | Regression label valid mask |",
        f"| `mom_mask_u8.mmap` | uint8 | `[T, N]` | Classification label valid mask |",
        f"| `both_mask_u8.mmap` | uint8 | `[T, N]` | Both labels valid mask |",
        "",
    ]

    path = out_dir / "COLUMN_REFERENCE.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Column reference → %s", path)


def _write_view_report(
    out_dir: Path,
    meta: dict,
    X_mm, y_ret_mm, y_mom_mm,
    ret_mask_mm, mom_mask_mm, both_mask_mm,
    build_time: float,
) -> None:
    """Write VIEW_REPORT.md with quality statistics about the built memmap."""
    T, N, D = meta["T"], meta["N"], meta["D"]
    F = meta["F"]
    dates = meta["dates"]
    total_cells = T * N

    # row_present stats
    rp = np.array(X_mm[:, :, D - 1], dtype=np.float32)
    n_present = int((rp > 0).sum())
    present_pct = n_present / total_cells * 100

    # Per-day present count
    daily_present = (rp > 0).sum(axis=1).astype(np.float64)
    avg_daily = float(daily_present.mean())
    min_daily = int(daily_present.min())
    max_daily = int(daily_present.max())

    # Feature NaN rate (from isna flags, sampled from first isna dim)
    # Average across all isna dims for present rows
    isna_sample = np.array(X_mm[:, :, F], dtype=np.float32)  # first isna flag
    nan_rate_overall = float(isna_sample[rp > 0].mean()) * 100 if n_present > 0 else 0.0

    # Label coverage
    ret_valid_total = int(ret_mask_mm.sum())
    mom_valid_total = int(mom_mask_mm.sum())
    both_valid_total = int(both_mask_mm.sum())

    ret_pct = ret_valid_total / total_cells * 100
    mom_pct = mom_valid_total / total_cells * 100
    both_pct = both_valid_total / total_cells * 100

    # Classification label distribution (among valid)
    if mom_valid_total > 0:
        mom_vals = np.array(y_mom_mm, dtype=np.int8)
        mom_valid_mask = np.array(mom_mask_mm, dtype=bool)
        valid_mom = mom_vals[mom_valid_mask]
        cls_counts = {}
        for c in range(5):
            cnt = int((valid_mom == c).sum())
            cls_counts[c] = cnt
    else:
        cls_counts = {}

    # Regression label stats (among valid)
    if ret_valid_total > 0:
        ret_vals = np.array(y_ret_mm, dtype=np.float32)
        ret_valid_mask = np.array(ret_mask_mm, dtype=bool)
        valid_ret = ret_vals[ret_valid_mask]
        ret_mean = float(np.mean(valid_ret))
        ret_std = float(np.std(valid_ret))
        ret_min = float(np.min(valid_ret))
        ret_max = float(np.max(valid_ret))
        ret_median = float(np.median(valid_ret))
    else:
        ret_mean = ret_std = ret_min = ret_max = ret_median = 0.0

    # Per-year summary
    year_stats = {}
    for t_idx, d in enumerate(dates):
        year = d[:4]
        if year not in year_stats:
            year_stats[year] = {"days": 0, "present": 0, "ret_valid": 0, "mom_valid": 0}
        year_stats[year]["days"] += 1
        year_stats[year]["present"] += int(daily_present[t_idx])
        year_stats[year]["ret_valid"] += int(ret_mask_mm[t_idx].sum())
        year_stats[year]["mom_valid"] += int(mom_mask_mm[t_idx].sum())

    # Build report
    lines = [
        "# View Report (lstm_mtl)",
        "",
        f"> Auto-generated by `build_view.py`. Build time: {build_time:.1f}s",
        "",
        "## 1. Basic Info",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Trading days (T) | {T:,} |",
        f"| Instruments (N) | {N:,} |",
        f"| Input dim (D) | {D} (2×{F} features + 1 row_present) |",
        f"| Date range | {dates[0]} → {dates[-1]} |",
        f"| Total cells (T×N) | {total_cells:,} |",
        f"| Build time | {build_time:.1f}s |",
        "",
        "## 2. Row Present (sample usability)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Present cells | {n_present:,} / {total_cells:,} ({present_pct:.1f}%) |",
        f"| Avg stocks/day | {avg_daily:.0f} |",
        f"| Min stocks/day | {min_daily:,} |",
        f"| Max stocks/day | {max_daily:,} |",
        "",
        "## 3. Feature Quality",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Feature NaN rate (present rows, sampled) | {nan_rate_overall:.2f}% |",
        "",
        "## 4. Label Coverage",
        "",
        "| Label | Valid cells | Coverage |",
        "|---|---|---|",
        f"| Regression (`{meta['regression_label']}`) | {ret_valid_total:,} | {ret_pct:.1f}% |",
        f"| Classification (`{meta['classification_label']}`) | {mom_valid_total:,} | {mom_pct:.1f}% |",
        f"| Both valid | {both_valid_total:,} | {both_pct:.1f}% |",
        "",
        "## 5. Regression Label Statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean | {ret_mean:.6f} |",
        f"| Std | {ret_std:.6f} |",
        f"| Min | {ret_min:.6f} |",
        f"| Median | {ret_median:.6f} |",
        f"| Max | {ret_max:.6f} |",
        "",
        "## 6. Classification Label Distribution",
        "",
        "| Class | Count | Proportion |",
        "|---|---|---|",
    ]
    for c in range(5):
        cnt = cls_counts.get(c, 0)
        pct = cnt / mom_valid_total * 100 if mom_valid_total > 0 else 0
        lines.append(f"| {c} | {cnt:,} | {pct:.1f}% |")

    lines += [
        "",
        "## 7. Per-Year Summary",
        "",
        "| Year | Days | Avg present/day | Ret valid | Mom valid |",
        "|---|---|---|---|---|",
    ]
    for year in sorted(year_stats):
        ys = year_stats[year]
        avg_p = ys["present"] / ys["days"] if ys["days"] > 0 else 0
        lines.append(
            f"| {year} | {ys['days']} | {avg_p:.0f} | {ys['ret_valid']:,} | {ys['mom_valid']:,} |"
        )

    lines.append("")

    path = out_dir / "VIEW_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("View report → %s", path)


# ======================================================================
# Dataset (adapted from reference 2_dataset_memmap.py)
# ======================================================================

class MemmapDayWindowDataset(Dataset):
    """
    Memory-mapped sliding window dataset for LSTM training.

    Each sample corresponds to one trading day and returns K randomly
    sampled stocks, each with a lookback window of L time-steps.

    Returns
    -------
    dict with keys:
        X         : [K, L, D] float32  — feature tensor
        y_ret     : [K]       float32  — regression target
        y_mom     : [K]       int64    — classification target
        ret_mask  : [K]       float32  — regression validity
        mom_mask  : [K]       float32  — classification validity
        both_mask : [K]       float32  — both valid
        date_idx  : int                — memmap date index
    """

    def __init__(
        self,
        memmap_dir: str | Path,
        lookback: int,
        day_start: int,
        day_end: int,
        k: int,
        seed: int = 42,
        sample_present_only: bool = True,
        feature_indices: Optional[Sequence[int]] = None,
    ):
        self.memmap_dir = str(memmap_dir)
        self.lookback = lookback
        self.k = k
        self.sample_present_only = sample_present_only
        self.rng = np.random.RandomState(seed)
        self.feature_indices = feature_indices

        assert day_start >= lookback - 1, (
            f"day_start={day_start} must >= lookback-1={lookback - 1}"
        )
        self.day_idxs = np.arange(day_start, day_end + 1, dtype=np.int32)

        # Load metadata
        with open(os.path.join(self.memmap_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.T = meta["T"]
        self.N = meta["N"]
        self.D = meta["D"]
        self.dates = meta["dates"]
        self.instruments = meta["instruments"]
        self.feature_cols = meta.get("feat_cols", [])

        self._row_present_idx = self.D - 1

        # Lazy-open memmap files (for Windows multiprocessing compatibility)
        self._X = None
        self._yret = None
        self._ymom = None
        self._rm = None
        self._mm = None
        self._bm = None

    def _lazy_open(self):
        if self._X is not None:
            return

        def mm(name, dtype, shape):
            return np.memmap(
                os.path.join(self.memmap_dir, name),
                dtype=dtype, mode="r", shape=shape,
            )

        self._X = mm("X_f16.mmap", np.float16, (self.T, self.N, self.D))
        self._yret = mm("y_ret_f32.mmap", np.float32, (self.T, self.N))
        self._ymom = mm("y_mom_i8.mmap", np.int8, (self.T, self.N))
        self._rm = mm("ret_mask_u8.mmap", np.uint8, (self.T, self.N))
        self._mm = mm("mom_mask_u8.mmap", np.uint8, (self.T, self.N))
        self._bm = mm("both_mask_u8.mmap", np.uint8, (self.T, self.N))

    def __len__(self):
        return len(self.day_idxs)

    def __getitem__(self, i):
        self._lazy_open()

        t = int(self.day_idxs[i])
        L = self.lookback

        # Sampling strategy: prefer stocks with row_present=1
        if self.sample_present_only:
            present = self._X[t, :, self._row_present_idx]
            valid_idx = np.flatnonzero(present > 0)

            if len(valid_idx) >= self.k:
                idx = self.rng.choice(valid_idx, size=self.k, replace=False)
            elif len(valid_idx) > 0:
                other_idx = np.setdiff1d(np.arange(self.N), valid_idx)
                n_other = self.k - len(valid_idx)
                other_sample = self.rng.choice(
                    other_idx, size=min(n_other, len(other_idx)), replace=False
                )
                idx = np.concatenate([valid_idx, other_sample])
                if len(idx) < self.k:
                    idx = np.concatenate([
                        idx,
                        self.rng.choice(self.N, size=self.k - len(idx), replace=False),
                    ])
            else:
                idx = self.rng.choice(self.N, size=self.k, replace=False)
        else:
            idx = self.rng.choice(self.N, size=self.k, replace=False)

        # Slice lookback window: [t-L+1 : t+1, idx, :] -> [L, K, D] -> [K, L, D]
        x = self._X[t - L + 1: t + 1, idx, :]
        x = np.transpose(x, (1, 0, 2)).astype(np.float32, copy=False)

        if self.feature_indices is not None:
            x = x[:, :, self.feature_indices]

        y_ret = self._yret[t, idx].astype(np.float32, copy=False)
        y_mom = self._ymom[t, idx].astype(np.int64, copy=False)
        ret_mask = self._rm[t, idx].astype(np.float32, copy=False)
        mom_mask = self._mm[t, idx].astype(np.float32, copy=False)
        both_mask = self._bm[t, idx].astype(np.float32, copy=False)

        return {
            "X": torch.from_numpy(x),
            "y_ret": torch.from_numpy(y_ret),
            "y_mom": torch.from_numpy(y_mom),
            "ret_mask": torch.from_numpy(ret_mask),
            "mom_mask": torch.from_numpy(mom_mask),
            "both_mask": torch.from_numpy(both_mask),
            "date_idx": t,
        }


# ======================================================================
# LSTMViewBuilder
# ======================================================================

class LSTMViewBuilder(BaseViewBuilder):
    """View builder for the LSTM multi-task model."""

    name = "lstm_mtl"

    def required_columns(self) -> List[str]:
        return []  # dynamic — validated at build time

    @staticmethod
    def compute_feature_indices(
        feat_cols: List[str],
        F: int,
        D: int,
        include_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
    ) -> Optional[List[int]]:
        """
        Compute which dimensions of X[t, n, :] to keep based on feature name filters.

        The memmap layout is [features(F), isna_flags(F), row_present(1)].
        When a feature at index i is selected, its isna flag at index F+i is also selected.
        row_present (index D-1) is always included.

        Returns None if no filtering is needed (all features selected).
        """
        import re

        selected = list(range(F))  # start with all features

        if include_pattern:
            pat = re.compile(include_pattern)
            selected = [i for i in selected if pat.search(feat_cols[i])]

        if exclude_pattern:
            pat = re.compile(exclude_pattern)
            selected = [i for i in selected if not pat.search(feat_cols[i])]

        if len(selected) == F:
            return None  # no filtering needed

        # Build indices: selected features + their isna flags + row_present
        indices = selected + [F + i for i in selected] + [D - 1]
        return indices

    def resolve_feature_filter(self, meta: dict, config: dict):
        """
        Read config and compute (feature_indices, effective_input_dim).

        Reads from ``training`` (training-time filtering, no rebuild needed).
        Returns (None, D) if no filtering, or (indices_list, filtered_D).
        """
        model_train_cfg = config.get("training", {})
        include_pattern = model_train_cfg.get("include_pattern", None)
        exclude_pattern = model_train_cfg.get("exclude_pattern", None)

        F = meta["F"]
        D = meta["D"]
        feat_cols = meta["feat_cols"]

        indices = self.compute_feature_indices(
            feat_cols, F, D,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
        )

        if indices is None:
            return None, D

        # effective D = selected_features + their isna_flags + row_present
        n_selected = (len(indices) - 1) // 2  # exclude row_present
        effective_D = 2 * n_selected + 1
        logger.info(
            "Feature filter: %d / %d features selected (include=%s, exclude=%s) → D=%d",
            n_selected, F, include_pattern, exclude_pattern, effective_D,
        )
        return indices, effective_D

    def get_effective_input_dim(self, meta: dict, config: dict) -> int:
        """Return the effective input dimension after feature filtering."""
        _, effective_D = self.resolve_feature_filter(meta, config)
        return effective_D

    def build(
        self,
        paths: PathManager,
        config: dict,
    ) -> Path:
        """Build memmap panel from panel_base + target_blocks. Returns the memmap directory."""
        label_roles = config.get("model", {}).get("label_roles", {})

        reg_label = label_roles.get("regression", "return_c0c1")
        cls_label = label_roles.get("classification", "momentum_cls")

        memmap_out = paths.memmap_dir(self.name)

        # Check if already built
        meta_path = memmap_out / "meta.json"
        if meta_path.exists():
            logger.info("Memmap already exists at %s — skipping build", memmap_out)
            return memmap_out

        logger.info("Building memmap panel → %s", memmap_out)
        build_memmap_panel(
            paths=paths,
            out_dir=memmap_out,
            regression_label=reg_label,
            classification_label=cls_label,
            config=config,
        )
        return memmap_out

    def get_dataset(
        self,
        view_dir: Path,
        day_start: int,
        day_end: int,
        config: dict,
    ) -> MemmapDayWindowDataset:
        view_cfg = config.get("model", {}).get("view", {})
        lookback = view_cfg.get("lookback", 60)
        k = view_cfg.get("k", 512)
        seed = config.get("training", {}).get("seed", 42)

        # Resolve feature filtering from config
        meta_path = Path(view_dir) / "meta.json"
        feature_indices = None
        if meta_path.exists():
            import json as _json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = _json.load(f)
            feature_indices, _ = self.resolve_feature_filter(meta, config)

        return MemmapDayWindowDataset(
            memmap_dir=view_dir,
            lookback=lookback,
            day_start=day_start,
            day_end=day_end,
            k=k,
            seed=seed,
            sample_present_only=True,
            feature_indices=feature_indices,
        )

    def build_dataloader(
        self,
        dataset: MemmapDayWindowDataset,
        config: dict,
        shuffle: bool = True,
    ) -> DataLoader:
        train_cfg = config.get("training", {})
        batch_size = train_cfg.get("batch_size", 4)
        num_workers = train_cfg.get("num_workers", 0)
        pin_memory = train_cfg.get("pin_memory", False)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
        )
