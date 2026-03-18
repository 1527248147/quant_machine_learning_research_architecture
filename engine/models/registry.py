"""
Model registry.

Maps model names to (model_class, view_builder_class, trainer_class) triples
so that the training dispatcher can instantiate all three from a single config key.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Type

from engine.models.base import BaseModel
from engine.training.base import BaseTrainer
from engine.views.base import BaseViewBuilder

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Tuple[Type[BaseModel], Type[BaseViewBuilder], Type[BaseTrainer]]] = {}


def register_model(
    name: str,
    model_class: Type[BaseModel],
    view_class: Type[BaseViewBuilder],
    trainer_class: Type[BaseTrainer],
) -> None:
    _REGISTRY[name] = (model_class, view_class, trainer_class)
    logger.debug("Registered model '%s'", name)


def get_model_class(name: str) -> Type[BaseModel]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Model '{name}' not registered. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name][0]


def get_view_class(name: str) -> Type[BaseViewBuilder]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Model '{name}' not registered. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name][1]


def get_trainer_class(name: str) -> Type[BaseTrainer]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Model '{name}' not registered. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name][2]


def list_models() -> List[str]:
    return list(_REGISTRY.keys())


# -----------------------------------------------------------------------
# Built-in registrations
# -----------------------------------------------------------------------

def _register_builtins() -> None:
    from engine.models.impl.lstm_mtl import LSTMMTLModel
    from engine.training.impl.lstm_mtl.trainer import LSTMMTLTrainer
    from engine.views.impl.lstm_mtl import LSTMViewBuilder

    register_model("lstm_mtl", LSTMMTLModel, LSTMViewBuilder, LSTMMTLTrainer)

    from engine.models.impl.lgbm import LGBMRankModel
    from engine.training.impl.lgbm.trainer import LGBMRankTrainer
    from engine.views.impl.lgbm import LGBMViewBuilder

    register_model("lgbm_rank", LGBMRankModel, LGBMViewBuilder, LGBMRankTrainer)


_register_builtins()
