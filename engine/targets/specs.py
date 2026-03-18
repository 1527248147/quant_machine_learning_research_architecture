"""
TargetSpec: metadata describing a registered target.

Used by model-label contract validation (Phase 4) and by the registry.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TargetSpec:
    name: str
    task_type: str          # "regression", "classification", "ranking"
    family: Optional[str] = None   # e.g. "return", "momentum"
    horizon: Optional[int] = None  # forward window in trading days
    dtype: str = "float32"
    params: Dict = field(default_factory=dict)
