"""
LightGBM LambdaRank Trainer.

Implements the full training pipeline for LightGBM LambdaRank:
    - Per-day relevance binning (converting raw returns to ordinal labels)
    - Date-based grouping for LambdaRank
    - Custom RankIC evaluation metric
    - Selection (train on train, select on valid via early stopping)
    - Optional refit on train+valid
    - Test evaluation

Reference: project_alpha158+ricequant_fin+lgbm/train_models/
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore

from engine.io.paths import PathManager
from engine.training.base import BaseTrainer
from engine.training.results import (
    SelectionResult,
    TestResult,
    TrainingRunResult,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Grouping & Relevance utilities
# ======================================================================

def sort_and_group(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    sids: np.ndarray,
    feat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Sort by (date, sid) and compute group sizes for LambdaRank.

    Returns (X_sorted, y_sorted, dates_sorted, sids_sorted, group_sizes).
    """
    # Sort by date then sid
    order = np.lexsort((sids, dates))
    X_s = X[order]
    y_s = y[order]
    dates_s = dates[order]
    sids_s = sids[order]

    # Compute group sizes (consecutive runs of the same date)
    groups = []
    n = len(dates_s)
    i = 0
    while i < n:
        j = i + 1
        while j < n and dates_s[j] == dates_s[i]:
            j += 1
        groups.append(j - i)
        i = j

    return X_s, y_s, dates_s, sids_s, groups


def drop_small_groups(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    sids: np.ndarray,
    groups: List[int],
    min_group_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Drop groups (trading days) with fewer than min_group_size stocks."""
    if min_group_size <= 1:
        return X, y, dates, sids, groups

    keep_mask = np.zeros(len(X), dtype=bool)
    new_groups = []
    start = 0
    for size in groups:
        end = start + size
        if size >= min_group_size:
            keep_mask[start:end] = True
            new_groups.append(size)
        start = end

    return X[keep_mask], y[keep_mask], dates[keep_mask], sids[keep_mask], new_groups


def make_relevance_per_day(
    y: np.ndarray,
    group_sizes: List[int],
    n_bins: int = 10,
) -> np.ndarray:
    """
    Convert raw returns to per-day relevance bins for LambdaRank.

    Within each trading day, ranks the returns and maps them to
    ordinal bins [0, n_bins-1]. This is the label LightGBM sees.
    """
    rel = np.zeros_like(y, dtype=np.float32)
    start = 0
    for g in group_sizes:
        end = start + g
        yg = y[start:end]

        valid = np.isfinite(yg)
        if valid.sum() <= 1:
            start = end
            continue

        y_valid = yg[valid]
        if np.all(y_valid == y_valid[0]):
            start = end
            continue

        # Rank within day
        ranks = pd.Series(y_valid).rank(method="average").to_numpy()
        pct = (ranks - 1.0) / max(1.0, (y_valid.size - 1.0))
        bins = np.floor(pct * n_bins).astype(np.int32)
        bins = np.clip(bins, 0, n_bins - 1)

        out = np.zeros(g, dtype=np.float32)
        out[valid] = bins.astype(np.float32)
        rel[start:end] = out

        start = end

    return rel


def spearman_rankic_by_group(
    y: np.ndarray,
    preds: np.ndarray,
    group_sizes: List[int],
) -> Tuple[float, float, float, int]:
    """
    Compute per-day Spearman RankIC, then aggregate.

    Returns (mean, std, IR, n_days).
    """
    ics = []
    start = 0
    for g in group_sizes:
        end = start + g
        yg = y[start:end]
        pg = preds[start:end]
        start = end

        if yg.size < 3:
            continue

        valid = np.isfinite(yg) & np.isfinite(pg)
        yv = yg[valid]
        pv = pg[valid]
        if yv.size < 3:
            continue

        ry = pd.Series(yv).rank(method="average").to_numpy()
        rp = pd.Series(pv).rank(method="average").to_numpy()
        if np.std(ry) == 0 or np.std(rp) == 0:
            continue

        ic = float(np.corrcoef(ry, rp)[0, 1])
        if np.isfinite(ic):
            ics.append(ic)

    if not ics:
        return float("nan"), float("nan"), float("nan"), 0

    ics = np.array(ics, dtype=np.float64)
    mean = float(np.mean(ics))
    std = float(np.std(ics, ddof=1)) if ics.size > 1 else float("nan")
    ir = float(mean / std) if np.isfinite(std) and std > 0 else float("nan")
    return mean, std, ir, int(ics.size)


def make_feval_rankic(dset_info: dict):
    """Create a LightGBM-compatible feval function for RankIC."""
    def _feval(preds: np.ndarray, dataset: lgb.Dataset):
        info = dset_info.get(id(dataset))
        if info is None:
            return ("rank_ic_mean", float("nan"), True)
        raw_y, group_sizes = info
        mean, std, ir, n_days = spearman_rankic_by_group(raw_y, preds, group_sizes)
        return ("rank_ic_mean", mean, True)
    return _feval


# ======================================================================
# LGBMRankTrainer
# ======================================================================

class LGBMRankTrainer(BaseTrainer):
    """Trainer for the LightGBM LambdaRank model."""

    name = "lgbm_rank"

    def run(
        self,
        config: dict,
        paths: PathManager,
    ) -> TrainingRunResult:
        """
        Full LightGBM LambdaRank training pipeline.

        1. Preflight: validate contract
        2. Locate pre-built tabular view
        3. Split: date-based train/valid/test
        4. Prepare: group, filter, relevance binning
        5. Train with early stopping on valid RankIC
        6. Evaluate on train/valid/test
        7. Save model + summary
        """
        if lgb is None:
            raise ImportError("lightgbm is required: pip install lightgbm")

        train_cfg = config.get("training", {})
        eval_cfg = config.get("evaluation", {})
        model_cfg = config.get("model", {})

        model_name = model_cfg.get("name", "lgbm_rank")
        seed = train_cfg.get("seed", 42)
        run_test = eval_cfg.get("run_test", True)
        refit_before_test = eval_cfg.get("refit_before_test", False)

        self.set_seed(seed)

        # --- Experiment dir ---
        exp_name, run_dir, ckpt_dir, log_path = self.setup_experiment(
            config, paths, model_name,
        )

        # --- Resolve model + view classes ---
        from engine.models.registry import get_model_class, get_view_class

        ModelClass = get_model_class(model_name)
        ViewClass = get_view_class(model_name)

        model_wrapper = ModelClass()
        view_builder = ViewClass()

        # --- Preflight ---
        self.run_preflight(model_name, model_wrapper.contract, config)

        # --- Locate pre-built view ---
        view_dir = paths.memmap_dir(view_builder.name)
        meta = self.load_view_meta(view_dir)
        logger.info("Using pre-built tabular view at %s", view_dir)

        # --- Split ---
        lookback = 0  # no lookback needed for tabular models
        split = self.build_split(meta, config, lookback=lookback)

        # --- Load datasets ---
        logger.info("Loading train/valid datasets...")
        train_ds = view_builder.get_dataset(
            view_dir, split.train_start_idx, split.train_end_idx, config,
        )
        valid_ds = view_builder.get_dataset(
            view_dir, split.valid_start_idx, split.valid_end_idx, config,
        )

        # --- Prepare train/valid data ---
        lgbm_cfg = train_cfg
        min_group_size = lgbm_cfg.get("min_group_size", 30)
        missing_drop_thresh = lgbm_cfg.get("missing_drop_thresh", 0.98)
        exclude_regex = lgbm_cfg.get("exclude_regex", None)
        n_bins = lgbm_cfg.get("relevance_bins", 10)
        clip_y_abs = lgbm_cfg.get("clip_y_abs", 0.0)

        X_tr, y_tr_raw, dates_tr, sids_tr, feat_cols = train_ds.get_usable_data(
            min_group_size=1,
            missing_drop_thresh=missing_drop_thresh,
            exclude_regex=exclude_regex,
        )
        X_va, y_va_raw, dates_va, sids_va, _ = valid_ds.get_usable_data(
            min_group_size=1,
            missing_drop_thresh=1.0,  # don't drop features based on valid
            exclude_regex=exclude_regex,
        )
        # Align valid to same feature set as train
        if len(feat_cols) < valid_ds.X.shape[1]:
            feat_idx = [valid_ds.feat_cols.index(c) for c in feat_cols if c in valid_ds.feat_cols]
            # Re-filter valid to match train features
            va_mask = (valid_ds.y_valid > 0) & (valid_ds.row_usable > 0)
            X_va = valid_ds.X[va_mask][:, feat_idx]
            y_va_raw = valid_ds.y[va_mask]
            dates_va = valid_ds.dates[va_mask]
            sids_va = valid_ds.sids[va_mask]

        # Clip labels
        if clip_y_abs and clip_y_abs > 0:
            y_tr_raw = np.clip(y_tr_raw, -clip_y_abs, clip_y_abs)
            y_va_raw = np.clip(y_va_raw, -clip_y_abs, clip_y_abs)

        # Sort and group
        X_tr, y_tr_raw, dates_tr, sids_tr, g_train = sort_and_group(
            X_tr, y_tr_raw, dates_tr, sids_tr, feat_cols,
        )
        X_va, y_va_raw, dates_va, sids_va, g_valid = sort_and_group(
            X_va, y_va_raw, dates_va, sids_va, feat_cols,
        )

        # Drop small groups
        X_tr, y_tr_raw, dates_tr, sids_tr, g_train = drop_small_groups(
            X_tr, y_tr_raw, dates_tr, sids_tr, g_train, min_group_size,
        )
        X_va, y_va_raw, dates_va, sids_va, g_valid = drop_small_groups(
            X_va, y_va_raw, dates_va, sids_va, g_valid, min_group_size,
        )

        if not g_train or not g_valid:
            raise RuntimeError("No groups left after min_group_size filtering.")

        logger.info(
            "Train: %d rows, %d days, %d features",
            len(X_tr), len(g_train), len(feat_cols),
        )
        logger.info(
            "Valid: %d rows, %d days",
            len(X_va), len(g_valid),
        )

        # Relevance binning
        y_tr_rel = make_relevance_per_day(y_tr_raw, g_train, n_bins=n_bins)
        y_va_rel = make_relevance_per_day(y_va_raw, g_valid, n_bins=n_bins)

        # Create LightGBM datasets
        dtrain = lgb.Dataset(
            X_tr, label=y_tr_rel, group=g_train,
            feature_name=feat_cols, free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_va, label=y_va_rel, group=g_valid,
            feature_name=feat_cols, free_raw_data=False,
        )

        # RankIC evaluation
        dset_info = {
            id(dtrain): (y_tr_raw, g_train),
            id(dvalid): (y_va_raw, g_valid),
        }

        # --- Train ---
        logger.info("=" * 60)
        logger.info("TRAINING LightGBM LambdaRank")
        t_train = time.time()

        model_wrapper.build_model(len(feat_cols), config)
        train_info = model_wrapper.fit(
            train_loader=dtrain,
            valid_loader=dvalid,
            config=config,
            callbacks={"feval": make_feval_rankic(dset_info)},
        )

        train_time = time.time() - t_train
        best_iter = train_info["best_iteration"]
        logger.info("Training complete in %.1f s (best_iteration=%d)", train_time, best_iter)

        # --- Evaluate ---
        p_tr = model_wrapper.predict(X_tr)["pred"]
        p_va = model_wrapper.predict(X_va)["pred"]

        tr_mean, tr_std, tr_ir, tr_days = spearman_rankic_by_group(y_tr_raw, p_tr, g_train)
        va_mean, va_std, va_ir, va_days = spearman_rankic_by_group(y_va_raw, p_va, g_valid)

        logger.info(
            "Train RankIC: mean=%.4f std=%.4f IR=%.4f (%d days)",
            tr_mean, tr_std, tr_ir, tr_days,
        )
        logger.info(
            "Valid RankIC: mean=%.4f std=%.4f IR=%.4f (%d days)",
            va_mean, va_std, va_ir, va_days,
        )

        # Save model
        model_path = ckpt_dir / "model.txt"
        model_wrapper.save(model_path)

        # Save used features
        self._save_feature_info(ckpt_dir, feat_cols)

        valid_metrics = {
            "rank_ic_mean": va_mean,
            "rank_ic_std": va_std,
            "rank_ic_ir": va_ir,
            "rank_ic_n_days": va_days,
        }

        selection = SelectionResult(
            best_epoch=1,
            best_iteration=best_iter,
            valid_metrics=valid_metrics,
            model_path=model_path,
        )

        # --- Test ---
        test_result = None
        if run_test:
            test_result = self._run_test(
                model_wrapper=model_wrapper,
                view_builder=view_builder,
                view_dir=view_dir,
                split=split,
                config=config,
                feat_cols=feat_cols,
                train_ds=train_ds,
                valid_ds=valid_ds,
                selection=selection,
                refit_before_test=refit_before_test,
                ckpt_dir=ckpt_dir,
                min_group_size=min_group_size,
                exclude_regex=exclude_regex,
                n_bins=n_bins,
                clip_y_abs=clip_y_abs,
            )

        # --- Save summary ---
        summary = {
            "model_name": model_name,
            "best_iteration": best_iter,
            "n_features": len(feat_cols),
            "train_time_s": round(train_time, 1),
            "metrics": {
                "train_rank_ic_mean": tr_mean,
                "train_rank_ic_std": tr_std,
                "train_rank_ic_ir": tr_ir,
                "train_rank_ic_n_days": tr_days,
                "valid_rank_ic_mean": va_mean,
                "valid_rank_ic_std": va_std,
                "valid_rank_ic_ir": va_ir,
                "valid_rank_ic_n_days": va_days,
            },
        }

        if test_result:
            summary["metrics"].update(test_result.test_metrics)

        summary_path = run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Summary saved → %s", summary_path)

        # Write CSV log (single row for tree-based models)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("best_iteration,train_rankic,valid_rankic,train_ir,valid_ir,train_time_s\n")
            f.write(
                f"{best_iter},{tr_mean:.6f},{va_mean:.6f},"
                f"{tr_ir:.4f},{va_ir:.4f},{train_time:.1f}\n"
            )

        result = TrainingRunResult(
            selection=selection,
            test=test_result,
            config=config,
            exp_name=exp_name,
            run_dir=run_dir,
        )

        logger.info("=" * 60)
        logger.info("Training run complete → %s", run_dir)
        return result

    def _run_test(
        self,
        model_wrapper,
        view_builder,
        view_dir: Path,
        split,
        config: dict,
        feat_cols: List[str],
        train_ds,
        valid_ds,
        selection: SelectionResult,
        refit_before_test: bool,
        ckpt_dir: Path,
        min_group_size: int,
        exclude_regex: Optional[str],
        n_bins: int,
        clip_y_abs: float,
    ) -> TestResult:
        """Run test evaluation, optionally with refit."""
        logger.info("=" * 60)

        model_source = "selection"

        if refit_before_test:
            logger.info("REFIT on train+valid before test...")
            model_source = "refit"
            model_wrapper, refit_path = self._refit(
                model_wrapper=model_wrapper,
                view_builder=view_builder,
                view_dir=view_dir,
                split=split,
                config=config,
                feat_cols=feat_cols,
                train_ds=train_ds,
                valid_ds=valid_ds,
                selection=selection,
                ckpt_dir=ckpt_dir,
                min_group_size=min_group_size,
                exclude_regex=exclude_regex,
                n_bins=n_bins,
                clip_y_abs=clip_y_abs,
            )

        logger.info("TEST EVALUATION (model_source=%s)", model_source)

        test_ds = view_builder.get_dataset(
            view_dir, split.test_start_idx, split.test_end_idx, config,
        )

        # Filter test to same features as train
        te_mask = (test_ds.y_valid > 0) & (test_ds.row_usable > 0)
        feat_idx = [test_ds.feat_cols.index(c) for c in feat_cols if c in test_ds.feat_cols]
        X_te = test_ds.X[te_mask][:, feat_idx]
        y_te_raw = test_ds.y[te_mask]
        dates_te = test_ds.dates[te_mask]
        sids_te = test_ds.sids[te_mask]

        if clip_y_abs and clip_y_abs > 0:
            y_te_raw = np.clip(y_te_raw, -clip_y_abs, clip_y_abs)

        X_te, y_te_raw, dates_te, sids_te, g_test = sort_and_group(
            X_te, y_te_raw, dates_te, sids_te, feat_cols,
        )
        X_te, y_te_raw, dates_te, sids_te, g_test = drop_small_groups(
            X_te, y_te_raw, dates_te, sids_te, g_test, min_group_size,
        )

        if not g_test:
            logger.warning("No test groups after filtering.")
            return TestResult(
                model_source=model_source,
                test_metrics={"test_rank_ic_mean": float("nan")},
            )

        p_te = model_wrapper.predict(X_te)["pred"]
        te_mean, te_std, te_ir, te_days = spearman_rankic_by_group(
            y_te_raw, p_te, g_test,
        )

        logger.info(
            "Test RankIC: mean=%.4f std=%.4f IR=%.4f (%d days)",
            te_mean, te_std, te_ir, te_days,
        )

        # Save test predictions
        out_pred = pd.DataFrame({
            "date": dates_te,
            "sid": sids_te,
            "y_raw": y_te_raw,
            "pred": p_te,
        })
        pred_path = ckpt_dir.parent / "test_predictions.csv"
        out_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
        logger.info("Test predictions saved → %s", pred_path)

        return TestResult(
            model_source=model_source,
            test_metrics={
                "test_rank_ic_mean": te_mean,
                "test_rank_ic_std": te_std,
                "test_rank_ic_ir": te_ir,
                "test_rank_ic_n_days": te_days,
            },
            model_path=selection.model_path,
        )

    def _refit(
        self,
        model_wrapper,
        view_builder,
        view_dir: Path,
        split,
        config: dict,
        feat_cols: List[str],
        train_ds,
        valid_ds,
        selection: SelectionResult,
        ckpt_dir: Path,
        min_group_size: int,
        exclude_regex: Optional[str],
        n_bins: int,
        clip_y_abs: float,
    ):
        """Retrain on train+valid with the same number of boosting rounds."""
        from engine.models.registry import get_model_class

        model_name = config.get("model", {}).get("name", "lgbm_rank")
        ModelClass = get_model_class(model_name)
        refit_model = ModelClass()

        # Combine train + valid data
        tv_ds = view_builder.get_dataset(
            view_dir,
            split.train_start_idx,
            split.valid_end_idx,
            config,
        )

        X_tv, y_tv_raw, dates_tv, sids_tv, _ = tv_ds.get_usable_data(
            min_group_size=1,
            missing_drop_thresh=1.0,
            exclude_regex=exclude_regex,
        )
        feat_idx = [tv_ds.feat_cols.index(c) for c in feat_cols if c in tv_ds.feat_cols]
        tv_mask = (tv_ds.y_valid > 0) & (tv_ds.row_usable > 0)
        X_tv = tv_ds.X[tv_mask][:, feat_idx]
        y_tv_raw = tv_ds.y[tv_mask]
        dates_tv = tv_ds.dates[tv_mask]
        sids_tv = tv_ds.sids[tv_mask]

        if clip_y_abs and clip_y_abs > 0:
            y_tv_raw = np.clip(y_tv_raw, -clip_y_abs, clip_y_abs)

        X_tv, y_tv_raw, dates_tv, sids_tv, g_tv = sort_and_group(
            X_tv, y_tv_raw, dates_tv, sids_tv, feat_cols,
        )
        X_tv, y_tv_raw, dates_tv, sids_tv, g_tv = drop_small_groups(
            X_tv, y_tv_raw, dates_tv, sids_tv, g_tv, min_group_size,
        )

        y_tv_rel = make_relevance_per_day(y_tv_raw, g_tv, n_bins=n_bins)

        dtrain_tv = lgb.Dataset(
            X_tv, label=y_tv_rel, group=g_tv,
            feature_name=feat_cols, free_raw_data=False,
        )

        # Use fixed num_boost_round from selection
        refit_config = dict(config)
        refit_config.setdefault("training", {})["num_boost_round"] = selection.best_iteration
        refit_config["training"]["early_stopping_rounds"] = 0  # no early stopping

        refit_model.build_model(len(feat_cols), refit_config)
        refit_model.fit(
            train_loader=dtrain_tv,
            valid_loader=dtrain_tv,  # dummy
            config=refit_config,
            callbacks={},
        )

        refit_path = ckpt_dir / "model_refit.txt"
        refit_model.save(refit_path)
        logger.info("Refit model saved → %s", refit_path)

        return refit_model, refit_path

    @staticmethod
    def _save_feature_info(ckpt_dir: Path, feat_cols: List[str]) -> None:
        """Save used features list."""
        feat_path = ckpt_dir / "used_features.txt"
        feat_path.write_text("\n".join(feat_cols) + "\n", encoding="utf-8")
        pd.DataFrame({"feature": feat_cols}).to_csv(
            ckpt_dir / "used_features.csv", index=False, encoding="utf-8-sig",
        )
        logger.info("Used features saved: %d features", len(feat_cols))
