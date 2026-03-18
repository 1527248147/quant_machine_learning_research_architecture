"""
Training dispatcher.

Looks up the model-specific Trainer from the registry and delegates to it.
All model-specific training logic lives in the model's own trainer subclass
(e.g., ``engine.training.impl.lstm_mtl.trainer.LSTMMTLTrainer``).
"""
from __future__ import annotations

import logging

from engine.io.paths import PathManager
from engine.training.results import TrainingRunResult

logger = logging.getLogger(__name__)


def run_training(
    config: dict,
    paths: PathManager,
) -> TrainingRunResult:
    """
    Entry point for training.

    Resolves the model-specific Trainer from the registry and calls its
    ``run()`` method.  This keeps the dispatcher model-agnostic.
    """
    from engine.models.registry import get_trainer_class

    model_name = config.get("model", {}).get("name", "lstm_mtl")
    logger.info("Training model '%s'", model_name)

    TrainerClass = get_trainer_class(model_name)
    trainer = TrainerClass()

    return trainer.run(config, paths)
