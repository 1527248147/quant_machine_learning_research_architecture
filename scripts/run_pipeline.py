"""
scripts/run_pipeline.py

一键运行完整 pipeline：assets → panel_base → targets → view → train

带断点续跑功能：每步成功后写 checkpoint，重跑时自动跳过已完成的步骤。
如果 config 变化，从受影响的最早步骤开始重跑。

Usage:
    # 全部从头跑（自动跳过已完成的步骤）
    python scripts/run_pipeline.py --config configs/default.yaml

    # 强制从头开始（忽略 checkpoint）
    python scripts/run_pipeline.py --config configs/default.yaml --restart

    # 从某一步开始跑（手动指定，忽略 checkpoint）
    python scripts/run_pipeline.py --config configs/default.yaml --start view

    # 只跑到某一步为止
    python scripts/run_pipeline.py --config configs/default.yaml --end view

    # 强制重建 view（即使已存在）
    python scripts/run_pipeline.py --config configs/default.yaml --start view --force

    # 查看会执行哪些步骤（不实际运行）
    python scripts/run_pipeline.py --config configs/default.yaml --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# Pipeline steps in order
STEPS = [
    ("assets",     "scripts/build_assets.py",      "构建基础数据 (instrument_master, trading_calendar, daily_bars, factors)"),
    ("panel_base", "scripts/build_panel_base.py",   "构建统一底板 panel_base.parquet"),
    ("targets",    "scripts/build_target.py",       "构建标签 target_block(s)"),
    ("view",       "scripts/build_view.py",         "构建模型视图 (memmap)"),
    ("train",      "scripts/train.py",              "训练模型"),
]

STEP_NAMES = [s[0] for s in STEPS]

# Checkpoint file location
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / ".pipeline_checkpoint.json"


# -----------------------------------------------------------------------
# Config hashing — detect config changes
# -----------------------------------------------------------------------
def _hash_config(config_path: str) -> str:
    """Compute SHA-256 of the config file content."""
    content = Path(config_path).read_bytes()
    return hashlib.sha256(content).hexdigest()


# -----------------------------------------------------------------------
# Checkpoint read / write
# -----------------------------------------------------------------------
def _load_checkpoint() -> dict:
    """
    Load checkpoint. Returns dict like:
    {
        "config_hash": "abc123...",
        "completed": ["assets", "panel_base", "targets"],
        "timestamps": {"assets": "2025-03-17 14:30:00", ...}
    }
    """
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_checkpoint(ckpt: dict) -> None:
    """Save checkpoint to disk."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps(ckpt, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _clear_checkpoint() -> None:
    """Remove checkpoint file."""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


def _mark_step_done(ckpt: dict, step_name: str, config_hash: str) -> dict:
    """Mark a step as completed in the checkpoint."""
    ckpt["config_hash"] = config_hash
    if "completed" not in ckpt:
        ckpt["completed"] = []
    if step_name not in ckpt["completed"]:
        ckpt["completed"].append(step_name)
    if "timestamps" not in ckpt:
        ckpt["timestamps"] = {}
    ckpt["timestamps"][step_name] = time.strftime("%Y-%m-%d %H:%M:%S")
    return ckpt


# -----------------------------------------------------------------------
# Resume logic
# -----------------------------------------------------------------------
def _find_resume_step(ckpt: dict, config_hash: str) -> int:
    """
    Determine which step index to resume from.

    - If config changed: restart from step 0
    - Otherwise: resume from first incomplete step
    """
    if not ckpt or "completed" not in ckpt:
        return 0

    if ckpt.get("config_hash") != config_hash:
        logger.info("Config 已变化，checkpoint 失效，从头开始")
        return 0

    completed = ckpt["completed"]

    # Find the first step that's NOT in completed
    for i, name in enumerate(STEP_NAMES):
        if name not in completed:
            return i

    # All steps completed
    return len(STEP_NAMES)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full quantitative research pipeline (with auto-resume)",
    )
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--start", choices=STEP_NAMES, default=None,
        help="Start from this step (overrides auto-resume)",
    )
    p.add_argument(
        "--end", choices=STEP_NAMES, default=STEP_NAMES[-1],
        help=f"Stop after this step (default: {STEP_NAMES[-1]})",
    )
    p.add_argument(
        "--restart", action="store_true",
        help="Ignore checkpoint, run from the beginning",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Force rebuild for steps that support it (e.g. view)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the steps that would be executed without running them",
    )
    return p.parse_args()


def run_step(name: str, script: str, description: str, config: str, force: bool) -> None:
    """Run a single pipeline step as a subprocess."""
    cmd = [sys.executable, script, "--config", config]

    if force and name == "view":
        cmd.append("--force")

    logger.info("=" * 60)
    logger.info("STEP: %s — %s", name, description)
    logger.info("CMD:  %s", " ".join(cmd))
    logger.info("=" * 60)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error("FAILED: %s (exit code %d, %.1fs)", name, result.returncode, elapsed)
        raise RuntimeError(f"Step '{name}' failed with exit code {result.returncode}")
    else:
        logger.info("DONE: %s (%.1fs)", name, elapsed)


def main() -> None:
    args = parse_args()
    config_hash = _hash_config(args.config)
    end_idx = STEP_NAMES.index(args.end)

    # --- Determine start index ---
    if args.restart:
        _clear_checkpoint()
        start_idx = 0
        logger.info("--restart: 清除 checkpoint，从头开始")
    elif args.start is not None:
        # Manual --start overrides auto-resume, also clear checkpoint
        # from that step onward
        start_idx = STEP_NAMES.index(args.start)
        ckpt = _load_checkpoint()
        if "completed" in ckpt:
            ckpt["completed"] = [s for s in ckpt["completed"]
                                 if STEP_NAMES.index(s) < start_idx]
            _save_checkpoint(ckpt)
        logger.info("--start %s: 从指定步骤开始", args.start)
    else:
        # Auto-resume from checkpoint
        ckpt = _load_checkpoint()
        start_idx = _find_resume_step(ckpt, config_hash)

        if start_idx >= len(STEP_NAMES) or start_idx > end_idx:
            logger.info("所有步骤已完成（checkpoint 有效）。如需重跑，请用 --restart")
            return

        if start_idx > 0:
            skipped = STEP_NAMES[:start_idx]
            logger.info("自动续跑：跳过已完成的步骤 %s", skipped)

    if start_idx > end_idx:
        logger.error("--start (%s) is after --end (%s)", STEP_NAMES[start_idx], args.end)
        sys.exit(1)

    steps_to_run = STEPS[start_idx:end_idx + 1]

    logger.info("Pipeline: %s → %s (%d steps)",
                steps_to_run[0][0], steps_to_run[-1][0], len(steps_to_run))
    for name, script, desc in steps_to_run:
        logger.info("  [%d] %s — %s", STEP_NAMES.index(name) + 1, name, desc)

    if args.dry_run:
        logger.info("(dry-run, not executing)")
        return

    # Load or init checkpoint
    ckpt = _load_checkpoint()
    if args.restart or ckpt.get("config_hash") != config_hash:
        ckpt = {"config_hash": config_hash, "completed": [], "timestamps": {}}

    pipeline_t0 = time.time()

    for name, script, desc in steps_to_run:
        try:
            run_step(name, script, desc, args.config, args.force)
        except RuntimeError:
            # Step failed — save progress so far, then exit
            _save_checkpoint(ckpt)
            logger.error("")
            logger.error("Pipeline 在步骤 '%s' 失败。已保存进度。", name)
            logger.error("修复后直接重跑即可自动从 '%s' 继续：", name)
            logger.error("  python scripts/run_pipeline.py --config %s", args.config)
            sys.exit(1)

        # Step succeeded — record in checkpoint
        ckpt = _mark_step_done(ckpt, name, config_hash)
        _save_checkpoint(ckpt)

    total = time.time() - pipeline_t0
    logger.info("=" * 60)
    logger.info("Pipeline complete! Total time: %.1fs (%.1f min)", total, total / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
