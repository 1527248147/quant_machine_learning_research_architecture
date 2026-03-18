"""
LSTM Multi-Task Model.

Adapted from the LSTM-Momentum reference project (model_lstm_mtl.py).

Architecture:
    Input [N, L, D] where D = 2F + 1 (features + isna_flags + row_present)
    → InputFeatureGating (learnable per-feature gates with L1 sparsity)
    → Linear projection D → embed_dim
    → LayerNorm + Dropout
    → LSTM backbone (num_layers, hidden_size)
    → LayerNorm + Dropout
    → Regression head → pred_ret [N]
    → Classification head → mom_logits [N, C]
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.models.base import BaseModel, ModelLabelContract


# ======================================================================
# Input Feature Gating
# ======================================================================

class InputFeatureGating(nn.Module):
    """
    Per-feature learnable gate: x' = sigmoid(logits) ⊙ x

    Paired deletion: the same gate controls both the raw feature and its
    corresponding isna flag, so when g_i → 0 both are suppressed.
    """

    def __init__(
        self,
        num_features: int,
        init_logit: float = 2.0,
        fixed_zero_idx: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_features = int(num_features)

        logits = torch.full(
            (self.num_features,), float(init_logit),
            dtype=torch.float32, device=device,
        )
        self.gate_logits = nn.Parameter(logits)

        fixed_mask = torch.zeros(self.num_features, dtype=torch.float32, device=device)
        if fixed_zero_idx is not None:
            for i in fixed_zero_idx:
                if 0 <= int(i) < self.num_features:
                    fixed_mask[int(i)] = 1.0
        self.register_buffer("fixed_zero_mask", fixed_mask, persistent=True)

    def gates(self) -> torch.Tensor:
        g = torch.sigmoid(self.gate_logits)
        if self.fixed_zero_mask is not None and self.fixed_zero_mask.sum() > 0:
            g = g * (1.0 - self.fixed_zero_mask)
        return g

    def forward(self, x_feat: torch.Tensor) -> torch.Tensor:
        return x_feat * self.gates()

    def l1(self, reduction: str = "mean") -> torch.Tensor:
        g = self.gates().abs()
        return g.sum() if reduction == "sum" else g.mean()

    def l1_logit(self, reduction: str = "mean") -> torch.Tensor:
        w = self.gate_logits.abs()
        if self.fixed_zero_mask is not None and self.fixed_zero_mask.sum() > 0:
            w = w * (1.0 - self.fixed_zero_mask)
        return w.sum() if reduction == "sum" else w.mean()


# ======================================================================
# Model config
# ======================================================================

@dataclass
class LSTMMTLConfig:
    input_dim: int          # D = 2F + 1
    raw_feature_dim: int    # F
    embed_dim: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    num_classes: int = 5
    use_layernorm: bool = True
    use_gating: bool = True
    gating_init_logit: float = 2.0


# ======================================================================
# nn.Module
# ======================================================================

class LSTMMultiTask(nn.Module):
    """LSTM backbone with regression + classification heads."""

    def __init__(
        self,
        cfg: LSTMMTLConfig,
        fixed_zero_idx: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.cfg = cfg

        if cfg.input_dim != 2 * cfg.raw_feature_dim + 1:
            raise ValueError(
                f"input_dim must equal 2*raw_feature_dim+1. "
                f"Got input_dim={cfg.input_dim}, raw_feature_dim={cfg.raw_feature_dim}"
            )

        self.raw_F = cfg.raw_feature_dim
        self.D = cfg.input_dim

        self.gating = None
        if cfg.use_gating:
            self.gating = InputFeatureGating(
                num_features=self.raw_F,
                init_logit=cfg.gating_init_logit,
                fixed_zero_idx=fixed_zero_idx,
            )

        self.input_proj = nn.Linear(self.D, cfg.embed_dim)
        self.in_ln = nn.LayerNorm(cfg.embed_dim) if cfg.use_layernorm else nn.Identity()
        self.in_drop = nn.Dropout(cfg.dropout)

        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=(cfg.dropout if cfg.num_layers > 1 else 0.0),
            bidirectional=False,
        )

        self.post_ln = nn.LayerNorm(cfg.hidden_size) if cfg.use_layernorm else nn.Identity()
        self.post_drop = nn.Dropout(cfg.dropout)

        self.ret_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, 1),
        )

        self.mom_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : [N, L, D]

        Returns
        -------
        pred_ret   : [N]
        mom_logits : [N, C]
        """
        feat = x[..., :self.raw_F]
        isna = x[..., self.raw_F: 2 * self.raw_F]
        rp = x[..., 2 * self.raw_F:]

        if self.gating is not None:
            feat = self.gating(feat)
            isna = self.gating(isna)

        x2 = torch.cat([feat, isna, rp], dim=-1)

        z = self.input_proj(x2)
        z = self.in_ln(z)
        z = self.in_drop(z)

        out, _ = self.lstm(z)
        h = out[:, -1, :]  # last time-step

        h = self.post_ln(h)
        h = self.post_drop(h)

        pred_ret = self.ret_head(h).squeeze(-1)
        mom_logits = self.mom_head(h)
        return pred_ret, mom_logits

    @torch.no_grad()
    def gate_values(self) -> Optional[torch.Tensor]:
        if self.gating is None:
            return None
        return self.gating.gates().detach().cpu()

    def gate_l1(self, reduction: str = "mean") -> torch.Tensor:
        if self.gating is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.gating.l1(reduction=reduction)


# ======================================================================
# BaseModel wrapper
# ======================================================================

class LSTMMTLModel(BaseModel):
    """Framework-compatible wrapper around LSTMMultiTask."""

    name = "lstm_mtl"
    contract = ModelLabelContract(
        mode="ROLE_BASED",
        min_labels=2,
        max_labels=2,
        required_roles={"regression": "regression", "classification": "classification"},
    )

    def __init__(self):
        self.model: Optional[LSTMMultiTask] = None
        self.cfg: Optional[LSTMMTLConfig] = None
        self.device = torch.device("cpu")
        self.feature_names: list = []

    def build_model(self, input_dim: int, config: dict) -> None:
        model_params = config.get("model", {}).get("params", {})

        raw_F = (input_dim - 1) // 2
        self.cfg = LSTMMTLConfig(
            input_dim=input_dim,
            raw_feature_dim=raw_F,
            embed_dim=model_params.get("embed_dim", 128),
            hidden_size=model_params.get("hidden_size", 128),
            num_layers=model_params.get("num_layers", 2),
            dropout=model_params.get("dropout", 0.2),
            num_classes=model_params.get("num_classes", 5),
            use_layernorm=model_params.get("use_layernorm", True),
            use_gating=model_params.get("use_gating", True),
            gating_init_logit=model_params.get("gating_init_logit", 2.0),
        )
        self.model = LSTMMultiTask(self.cfg)

    def fit(self, train_loader, valid_loader, config, callbacks=None):
        raise NotImplementedError("Use training/trainer.py for the training loop")

    def predict(self, loader) -> dict:
        raise NotImplementedError("Use training/evaluator.py for evaluation")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "cfg": asdict(self.cfg) if self.cfg else {},
                "feature_names": self.feature_names,
            },
            str(path),
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        if self.model is None and "cfg" in ckpt:
            self.cfg = LSTMMTLConfig(**ckpt["cfg"])
            self.model = LSTMMultiTask(self.cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.feature_names = ckpt.get("feature_names", [])
        self.model.to(self.device)

    def to_device(self, device) -> None:
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
