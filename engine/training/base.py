"""
Base class for all trainers.

Each model registers its own Trainer subclass. The trainer owns the full
training loop — including loss computation, evaluation metrics, and any
model-specific scheduling (e.g., gate L1 ramp for LSTM).

Shared utilities (seed, device, experiment dirs, split, early stopping)
are provided as helper methods so subclasses don't repeat boilerplate.
"""
from __future__ import annotations

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from engine.io.paths import PathManager
from engine.training.preflight import validate_contract
from engine.training.results import TrainingRunResult
from engine.training.splitter import SplitBundle, build_split_bundle

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base for all trainers."""

    name: str  # must match the model name

    # ------------------------------------------------------------------
    # Shared helpers (subclasses call these, don't override)
    # ------------------------------------------------------------------

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def resolve_device() -> torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        logger.info("Device: %s", device)
        return device

    @staticmethod
    def setup_experiment(
        config: dict,
        paths: PathManager,
        model_name: str,
    ) -> tuple:
        """
        Create experiment dirs and save config snapshot.

        Returns (exp_name, run_dir, ckpt_dir, log_path).
        """
        exp_name = config.get("experiment", {}).get(
            "name", f"{model_name}_{int(time.time())}"
        )
        run_dir = paths.run_dir(exp_name)
        ckpt_dir = paths.run_checkpoint_dir(exp_name)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_path = paths.run_log_path(exp_name)

        config_path = paths.run_config_path(exp_name)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=str)

        return exp_name, run_dir, ckpt_dir, log_path

    @staticmethod
    def run_preflight(model_name: str, contract, config: dict) -> None:
        """Validate model contract against config labels."""
        model_cfg = config.get("model", {})
        label_roles = model_cfg.get("label_roles", {})
        label_names = list(label_roles.values()) if label_roles else []
        if not label_names:
            label_names = [t["name"] for t in config.get("targets", [])]

        validate_contract(
            model_name=model_name,
            contract=contract,
            label_names=label_names,
            label_roles=label_roles if label_roles else None,
        )

    @staticmethod
    def load_view_meta(view_dir: Path) -> dict:
        """Load meta.json from a pre-built view directory."""
        meta_path = view_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Pre-built view not found at {view_dir}. "
                f"Run `python scripts/build_view.py --config <config>` first."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def build_split(meta: dict, config: dict, lookback: int) -> SplitBundle:
        """Build train/valid/test split from meta dates."""
        return build_split_bundle(meta["dates"], config, lookback=lookback)

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        config: dict,
        paths: PathManager,
    ) -> TrainingRunResult:
        """
        Full training pipeline.

        Subclasses implement the complete loop: build model, train, select,
        (optional) refit, (optional) test.  Use the shared helpers above
        for boilerplate.
        """
