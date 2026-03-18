"""
scripts/build_view.py

Step 6 of the pipeline: build model-specific view from panel_base + target_blocks.

For the LSTM model this converts the source parquets into a 3D memmap panel
[T, N, D] that the training loop can consume at high speed.

This step is independent of training — you can build the view once and
train multiple times without rebuilding.

Extensible: the script resolves the view builder from the model registry,
so adding a new model+view pair requires no changes here.

Usage:
    python scripts/build_view.py --config configs/default.yaml
    python scripts/build_view.py --config configs/default.yaml --force
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
from engine.models.registry import get_view_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_view")


def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build model-specific view from panel_base + target_blocks",
    )
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--force", action="store_true",
        help="Force rebuild even if memmap already exists",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)

    pm = PathManager(processed_dir=cfg["data"]["processed_dir"])
    model_name = cfg.get("model", {}).get("name", "lstm_mtl")

    # Resolve view builder from registry (extensible: any registered model works)
    ViewClass = get_view_class(model_name)
    view_builder = ViewClass()

    # Check if already exists
    view_dir = pm.memmap_dir(view_builder.name)
    meta_path = view_dir / "meta.json"
    if meta_path.exists() and not args.force:
        logger.info("View already exists at %s — use --force to rebuild", view_dir)
        return

    # If forcing, remove existing meta so build() won't skip
    if args.force and meta_path.exists():
        meta_path.unlink()
        logger.info("Removed existing meta.json for forced rebuild")

    logger.info("Model: %s → View: %s", model_name, view_builder.name)
    logger.info("panel_base: %s", pm.panel_base_path)

    # Build view (reads panel_base + target_blocks directly)
    out_dir = view_builder.build(pm, cfg)
    logger.info("View built → %s", out_dir)


if __name__ == "__main__":
    main()
