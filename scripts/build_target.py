"""
scripts/build_target.py

Step 3 of the pipeline: build target_block(s) from panel_base.

Memory-efficient: each recipe declares the columns it needs via
``required_columns()``, and the engine loads only those from the
parquet file (~120 MB instead of ~14 GB).

Usage:
    python scripts/build_target.py --config configs/default.yaml
    python scripts/build_target.py --config configs/default.yaml --target return_1d
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.io.paths import PathManager
from engine.io.parquet_io import load_parquet, save_parquet
from engine.targets.engine import build_target_block_from_path
from engine.targets.registry import list_targets
from engine.targets.report_generator import generate_target_report, generate_column_reference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_target")


def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_target_list(cfg: dict, override_target: str | None) -> list[dict]:
    """
    Parse target config into a list of {name, params} dicts.

    Supports:
      - --target CLI override (single target, params from config or registry defaults)
      - targets: [...]  (list of targets)
      - target: {name, params}  (legacy single target)
    """
    if override_target:
        for t in cfg.get("targets", []):
            if t["name"] == override_target:
                return [{"name": t["name"], "params": t.get("params", {})}]
        legacy = cfg.get("target", {})
        if legacy.get("name") == override_target:
            return [{"name": override_target, "params": legacy.get("params", {})}]
        return [{"name": override_target, "params": {}}]

    if "targets" in cfg:
        return [{"name": t["name"], "params": t.get("params", {})} for t in cfg["targets"]]

    legacy = cfg.get("target", {})
    name = legacy.get("name", "return_5d")
    return [{"name": name, "params": legacy.get("params", {})}]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build target_block(s) from panel_base")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--target",
        default=None,
        help="Build only this target (overrides config). e.g. return_1d",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)

    processed_dir = cfg["data"]["processed_dir"]
    pm = PathManager(processed_dir=processed_dir)

    target_list = _parse_target_list(cfg, args.target)
    logger.info("Targets to build: %s", [t["name"] for t in target_list])
    logger.info("Available registered targets: %s", list_targets())

    # Load trading calendar once (tiny file, shared across targets)
    logger.info("Loading trading_calendar from %s …", pm.trading_calendar_path)
    trading_calendar = load_parquet(pm.trading_calendar_path)

    # Build each target (engine loads only needed columns per recipe)
    for tgt in target_list:
        target_name = tgt["name"]
        recipe_params = tgt["params"]

        logger.info("=" * 60)

        t1 = time.time()
        target_block = build_target_block_from_path(
            pm.panel_base_path,
            target_name,
            trading_calendar=trading_calendar,
            **recipe_params,
        )
        elapsed = time.time() - t1
        logger.info("  target_block '%s' built in %.1f s", target_name, elapsed)

        # Save
        out_path = pm.target_block_path(target_name)
        pm.ensure_dir(out_path.parent)
        save_parquet(target_block, out_path, index=True)
        logger.info("  Saved → %s", out_path)

        # Quality report + column reference
        generate_target_report(target_block, target_name, out_path.parent)
        generate_column_reference(out_path, target_name, out_path.parent)

    logger.info("=" * 60)
    logger.info("All targets built: %s", [t["name"] for t in target_list])


if __name__ == "__main__":
    main()
