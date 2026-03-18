"""
Training callbacks: early stopping, gate lambda schedule, etc.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ======================================================================
# Early stopping
# ======================================================================

@dataclass
class EarlyStoppingState:
    best_val: float = float("inf")
    best_epoch: int = -1
    bad_epochs: int = 0
    should_stop: bool = False


class EarlyStopping:
    """Track validation loss and signal when to stop."""

    def __init__(self, patience: int, min_delta: float = 1e-9):
        self.patience = patience
        self.min_delta = min_delta
        self.state = EarlyStoppingState()

    def step(self, val_loss: float, epoch: int) -> EarlyStoppingState:
        if val_loss < self.state.best_val - self.min_delta:
            self.state.best_val = val_loss
            self.state.best_epoch = epoch
            self.state.bad_epochs = 0
        else:
            self.state.bad_epochs += 1

        self.state.should_stop = self.state.bad_epochs >= self.patience
        return self.state


# ======================================================================
# Gate lambda schedule
# ======================================================================

def gate_lambda_schedule(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    lam_max: float,
) -> float:
    """
    L1 gate coefficient schedule.

    Returns 0 during warmup, then linearly ramps to lam_max over ramp_epochs.
    """
    if lam_max <= 0:
        return 0.0
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return lam_max
    t = epoch - warmup_epochs
    if t >= ramp_epochs:
        return lam_max
    return lam_max * (t / ramp_epochs)
