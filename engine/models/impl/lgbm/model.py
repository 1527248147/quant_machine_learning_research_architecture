"""
LightGBM LambdaRank model for stock ranking.

Uses LightGBM's built-in LambdaRank objective to learn a ranking function
over stocks within each trading day. The model predicts a relevance score
that can be used to rank stocks cross-sectionally.

This is a tree-based model — no GPU required, no torch dependency at runtime.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore

from engine.models.base import BaseModel, ModelLabelContract

logger = logging.getLogger(__name__)


class LGBMRankModel(BaseModel):
    """LightGBM LambdaRank model wrapper."""

    name = "lgbm_rank"

    contract = ModelLabelContract(
        mode="ANY_SINGLE",
        min_labels=1,
        max_labels=1,
    )

    def __init__(self):
        self.booster: Optional[lgb.Booster] = None
        self.best_iteration: Optional[int] = None

    def build_model(self, input_dim: int, config: dict) -> None:
        """No-op for tree models — model is built during fit."""
        self.input_dim = input_dim
        logger.info("LGBMRankModel: input_dim=%d (model built during fit)", input_dim)

    def fit(
        self,
        train_loader: Any,
        valid_loader: Any,
        config: dict,
        callbacks: Optional[Dict] = None,
    ) -> dict:
        """
        Train LightGBM model.

        Parameters
        ----------
        train_loader : lgb.Dataset
        valid_loader : lgb.Dataset
        config : Full config dict
        callbacks : dict with optional keys:
            "feval": custom evaluation function
            "dset_info": mapping for RankIC evaluation

        Returns
        -------
        dict with training info (best_iteration, etc.)
        """
        if lgb is None:
            raise ImportError("lightgbm is required: pip install lightgbm")

        train_cfg = config.get("training", {})
        model_params = config.get("model", {}).get("params", {})

        num_boost_round = train_cfg.get("num_boost_round", 5000)
        early_stopping_rounds = train_cfg.get("early_stopping_rounds", 400)

        params = self._build_params(model_params, train_cfg)
        logger.info("LightGBM params: %s", params)

        lgb_callbacks = [lgb.log_evaluation(period=50)]
        if early_stopping_rounds > 0:
            lgb_callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    first_metric_only=True,
                    verbose=True,
                )
            )

        feval = callbacks.get("feval") if callbacks else None

        self.booster = lgb.train(
            params,
            train_loader,
            num_boost_round=num_boost_round,
            valid_sets=[valid_loader],
            valid_names=["valid"],
            feval=feval,
            callbacks=lgb_callbacks,
        )

        self.best_iteration = self.booster.best_iteration or num_boost_round
        logger.info("Training complete: best_iteration=%d", self.best_iteration)

        return {"best_iteration": self.best_iteration}

    def predict(self, loader: Any) -> dict:
        """
        Predict on feature matrix.

        Parameters
        ----------
        loader : np.ndarray of shape [n_samples, n_features]

        Returns
        -------
        dict with "pred" key containing prediction array.
        """
        if self.booster is None:
            raise RuntimeError("Model not trained yet.")
        preds = self.booster.predict(loader, num_iteration=self.best_iteration)
        return {"pred": preds}

    def save(self, path: Path) -> None:
        if self.booster is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        self.booster.save_model(str(path))
        logger.info("Model saved → %s", path)

    def load(self, path: Path) -> None:
        if lgb is None:
            raise ImportError("lightgbm is required: pip install lightgbm")
        path = Path(path)
        self.booster = lgb.Booster(model_file=str(path))
        logger.info("Model loaded ← %s", path)

    def to_device(self, device: Any) -> None:
        """No-op for tree models (CPU only)."""
        pass

    @staticmethod
    def _build_params(model_params: dict, train_cfg: dict) -> dict:
        """Build LightGBM parameter dict from config."""
        seed = train_cfg.get("seed", 42)
        n_bins = train_cfg.get("relevance_bins", 10)
        trunc_level = model_params.get("truncation_level", 50)

        params = {
            "objective": "lambdarank",
            "metric": "None",
            "learning_rate": model_params.get("learning_rate", 0.02),
            "max_depth": model_params.get("max_depth", 8),
            "num_leaves": model_params.get("num_leaves", 127),
            "min_data_in_leaf": model_params.get("min_data_in_leaf", 2000),
            "feature_fraction": model_params.get("feature_fraction", 0.7),
            "bagging_fraction": model_params.get("bagging_fraction", 0.7),
            "bagging_freq": model_params.get("bagging_freq", 1),
            "lambda_l1": model_params.get("lambda_l1", 0.0),
            "lambda_l2": model_params.get("lambda_l2", 10.0),
            "min_gain_to_split": model_params.get("min_gain_to_split", 0.02),
            "lambdarank_truncation_level": int(trunc_level),
            "label_gain": list(range(int(n_bins))),
            "verbosity": -1,
            "seed": int(seed),
            "num_threads": int(model_params.get("num_threads", 16)),
            "force_col_wise": True,
        }
        return params
