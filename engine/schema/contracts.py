"""
Contract definitions for the framework.

- ModelLabelContract: declares what labels a model requires.
  Used by training/preflight.py to validate config before training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class ModelLabelContract:
    """
    Declares what labels a model requires.

    Modes
    -----
    ANY_SINGLE  : Accepts exactly one label of any name.
    ANY_MULTI   : Accepts multiple labels of any name.
    EXACT       : Requires a specific set of label names.
    ROLE_BASED  : Requires labels mapped to named roles (e.g., regression, classification).
    CUSTOM      : Uses a user-provided validator callable.
    """
    mode: str  # "ANY_SINGLE" | "ANY_MULTI" | "EXACT" | "ROLE_BASED" | "CUSTOM"
    min_labels: int = 1
    max_labels: Optional[int] = 1
    exact_labels: List[str] = field(default_factory=list)
    required_roles: Dict[str, str] = field(default_factory=dict)
    custom_validator: Optional[Callable] = None
