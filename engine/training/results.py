"""
Training result dataclasses.

These are returned by the trainer to communicate selection, refit, and test outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SelectionResult:
    """Outcome of the train-on-train + select-on-valid stage."""
    best_epoch: int
    best_iteration: Optional[int]
    best_params: Dict = field(default_factory=dict)
    valid_metrics: Dict = field(default_factory=dict)
    model_path: Optional[Path] = None


@dataclass
class TestResult:
    """Outcome of the final test-set evaluation."""
    model_source: str  # "selection" or "refit"
    test_metrics: Dict = field(default_factory=dict)
    model_path: Optional[Path] = None


@dataclass
class TrainingRunResult:
    """Complete result of a training run."""
    selection: SelectionResult
    test: Optional[TestResult] = None  # None if run_test=false
    config: Dict = field(default_factory=dict)
    exp_name: str = ""
    run_dir: Optional[Path] = None
