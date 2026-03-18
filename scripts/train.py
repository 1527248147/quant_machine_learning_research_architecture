"""
scripts/train.py

Step 7 of the pipeline: train a model using pre-built memmap view.

Requires: memmap view already built via scripts/build_view.py.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --exp_name my_experiment
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.io.paths import PathManager
from engine.training.trainer import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a model using pre-built memmap view")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--exp_name",
        default=None,
        help="Experiment name (overrides config). Used for the run directory.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)

    if args.exp_name:
        cfg.setdefault("experiment", {})["name"] = args.exp_name

    pm = PathManager(processed_dir=cfg["data"]["processed_dir"])

    logger.info("Config: %s", args.config)

    result = run_training(config=cfg, paths=pm)

    logger.info("=" * 60)
    logger.info("Experiment: %s", result.exp_name)
    logger.info("Run dir: %s", result.run_dir)

    # Model-agnostic result display
    sel = result.selection
    vm = sel.valid_metrics
    if "loss" in vm:
        # LSTM-style metrics
        logger.info(
            "Selection: best_epoch=%d val_loss=%.6f",
            sel.best_epoch, vm["loss"],
        )
    elif "rank_ic_mean" in vm:
        # LightGBM-style metrics
        logger.info(
            "Selection: best_iteration=%s  Valid RankIC=%.4f (std=%.4f IR=%.4f)",
            sel.best_iteration,
            vm.get("rank_ic_mean", 0),
            vm.get("rank_ic_std", 0),
            vm.get("rank_ic_ir", 0),
        )
    else:
        logger.info("Selection: best_epoch=%d  metrics=%s", sel.best_epoch, vm)

    if result.test:
        tm = result.test.test_metrics
        if "loss" in tm:
            logger.info(
                "Test (%s): loss=%.6f IC=%.4f RankIC=%.4f Acc=%.4f",
                result.test.model_source,
                tm.get("loss", 0),
                tm.get("ic", 0),
                tm.get("rankic", 0),
                tm.get("cls_acc", 0),
            )
        elif "test_rank_ic_mean" in tm:
            logger.info(
                "Test (%s): RankIC=%.4f (std=%.4f IR=%.4f, %d days)",
                result.test.model_source,
                tm.get("test_rank_ic_mean", 0),
                tm.get("test_rank_ic_std", 0),
                tm.get("test_rank_ic_ir", 0),
                tm.get("test_rank_ic_n_days", 0),
            )
        else:
            logger.info("Test (%s): metrics=%s", result.test.model_source, tm)


if __name__ == "__main__":
    main()
