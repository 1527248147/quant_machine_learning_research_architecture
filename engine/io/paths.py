"""
Centralised path management for the framework.

All paths are resolved relative to a configurable ``processed_dir`` root.
The layout mirrors the data flow:

  data/
    raw/                  (raw inputs, user-provided)
    processed/
      assets/             (instrument_master, trading_calendar, daily_bars, …)
      panel/              (active_session_index, panel_base)
      targets/<name>/     (target_block per recipe)
      views/<name>/memmap/ (model-specific views, e.g. 3D memmap for LSTM)
    training_result/<exp_name>/  (train outputs: checkpoints, predictions, logs)
"""
from __future__ import annotations
import os
from pathlib import Path


class PathManager:
    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        training_result_dir: str | Path = "data/training_result",
    ):
        self.processed_dir = Path(processed_dir)
        self.training_result_dir = Path(training_result_dir)

    # -----------------------------------------------------------------------
    # Assets
    # -----------------------------------------------------------------------

    @property
    def assets_dir(self) -> Path:
        return self.processed_dir / "assets"

    @property
    def instrument_master_path(self) -> Path:
        return self.assets_dir / "instrument_master.parquet"

    @property
    def trading_calendar_path(self) -> Path:
        return self.assets_dir / "trading_calendar.parquet"

    @property
    def daily_bars_path(self) -> Path:
        return self.assets_dir / "daily_bars.parquet"

    @property
    def factor_values_path(self) -> Path:
        return self.assets_dir / "factor_values.parquet"

    @property
    def status_intervals_path(self) -> Path:
        return self.assets_dir / "status_intervals.parquet"

    # -----------------------------------------------------------------------
    # Panel
    # -----------------------------------------------------------------------

    @property
    def panel_dir(self) -> Path:
        return self.processed_dir / "panel"

    @property
    def active_session_index_path(self) -> Path:
        return self.panel_dir / "active_session_index.parquet"

    @property
    def panel_base_path(self) -> Path:
        return self.panel_dir / "panel_base.parquet"

    # -----------------------------------------------------------------------
    # Targets
    # -----------------------------------------------------------------------

    def target_dir(self, target_name: str) -> Path:
        return self.processed_dir / "targets" / target_name

    def target_block_path(self, target_name: str) -> Path:
        return self.target_dir(target_name) / "target_block.parquet"

    # -----------------------------------------------------------------------
    # Views / Memmap
    # -----------------------------------------------------------------------

    def view_dir(self, view_name: str) -> Path:
        return self.processed_dir / "views" / view_name

    def memmap_dir(self, view_name: str) -> Path:
        return self.view_dir(view_name) / "memmap"

    def memmap_meta_path(self, view_name: str) -> Path:
        return self.memmap_dir(view_name) / "meta.json"

    # -----------------------------------------------------------------------
    # Training Result
    # -----------------------------------------------------------------------

    def run_dir(self, exp_name: str) -> Path:
        return self.training_result_dir / exp_name

    def run_checkpoint_dir(self, exp_name: str) -> Path:
        return self.run_dir(exp_name) / "checkpoints"

    def run_log_path(self, exp_name: str) -> Path:
        return self.run_dir(exp_name) / "log.csv"

    def run_config_path(self, exp_name: str) -> Path:
        return self.run_dir(exp_name) / "config.json"

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def makedirs(self) -> None:
        """Create all standard output directories."""
        for p in [
            self.assets_dir,
            self.panel_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def ensure_dir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path
