"""
Target registry: maps target names to (recipe_class, spec) pairs.

Usage:
    from engine.targets.registry import get_recipe, get_spec, list_targets

Adding a new target:
    1. Create a recipe in engine/targets/recipes/<name>.py
    2. Register it here with register_target()
"""
from __future__ import annotations
from typing import Dict, List, Type

from engine.targets.base import BaseTargetRecipe
from engine.targets.specs import TargetSpec

_REGISTRY: Dict[str, dict] = {}


def register_target(
    name: str,
    recipe_class: Type[BaseTargetRecipe],
    spec: TargetSpec,
) -> None:
    """Register a target recipe and its spec."""
    _REGISTRY[name] = {"recipe_class": recipe_class, "spec": spec}


def get_recipe(target_name: str, **kwargs) -> BaseTargetRecipe:
    """Instantiate and return a recipe by target_name."""
    if target_name not in _REGISTRY:
        raise KeyError(
            f"Target '{target_name}' not registered. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[target_name]["recipe_class"](**kwargs)


def get_spec(name: str) -> TargetSpec:
    """Return the TargetSpec for a registered target."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Target '{name}' not registered. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]["spec"]


def list_targets() -> List[str]:
    """Return all registered target names."""
    return list(_REGISTRY.keys())


# -----------------------------------------------------------------------
# Auto-register built-in recipes on import
# -----------------------------------------------------------------------
def _register_builtins() -> None:
    from engine.targets.recipes.return_nd import ReturnRecipe
    from engine.targets.recipes.momentum_cls import MomentumClsRecipe

    # c0c1: 今日收盘 → 明日收盘（与 MiM-StocR 参考模型一致）
    register_target(
        "return_c0c1",
        ReturnRecipe,
        TargetSpec(
            name="return_c0c1",
            task_type="regression",
            family="return",
            horizon=1,
            dtype="float32",
            params={"formula": "c0c1"},
        ),
    )

    # o1c2: 次日开盘 → 第三日收盘（更贴近 A 股 T+1 实际可获得收益）
    register_target(
        "return_o1c2",
        ReturnRecipe,
        TargetSpec(
            name="return_o1c2",
            task_type="regression",
            family="return",
            horizon=2,
            dtype="float32",
            params={"formula": "o1c2"},
        ),
    )

    # c0c5: 今日收盘 → 5 日后收盘（5 日 close-to-close）
    register_target(
        "return_5d",
        ReturnRecipe,
        TargetSpec(
            name="return_5d",
            task_type="regression",
            family="return",
            horizon=5,
            dtype="float32",
            params={"formula": "c0c5"},
        ),
    )

    # o1c5: 次日开盘 → 5 日后收盘（约 1 周持仓）
    register_target(
        "return_o1c5",
        ReturnRecipe,
        TargetSpec(
            name="return_o1c5",
            task_type="regression",
            family="return",
            horizon=5,
            dtype="float32",
            params={"formula": "o1c5"},
        ),
    )

    # c0c20: 月度收益（20 交易日）
    register_target(
        "return_20d",
        ReturnRecipe,
        TargetSpec(
            name="return_20d",
            task_type="regression",
            family="return",
            horizon=20,
            dtype="float32",
            params={"formula": "c0c20"},
        ),
    )

    # c1c2: 次日收盘 → 第三日收盘（Qlib 风格: Ref($close,-2)/Ref($close,-1)-1）
    register_target(
        "return_c1c2",
        ReturnRecipe,
        TargetSpec(
            name="return_c1c2",
            task_type="regression",
            family="return",
            horizon=2,
            dtype="float32",
            params={"formula": "c1c2"},
        ),
    )

    register_target(
        "momentum_cls",
        MomentumClsRecipe,
        TargetSpec(
            name="momentum_cls",
            task_type="classification",
            family="momentum",
            horizon=7,
            dtype="int16",
            params={"gap": 4, "line_len": 7, "price_col": "market.close"},
        ),
    )


_register_builtins()
