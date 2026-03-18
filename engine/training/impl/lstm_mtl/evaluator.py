"""
Evaluation utilities for the LSTM multi-task model.

Provides IC, RankIC, accuracy metrics and a full eval_one_epoch function.
These are LSTM-specific — a different model (e.g., LightGBM) would have
its own evaluator with different metrics and forward-pass logic.

Adapted from the LSTM-Momentum reference project.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Correlation helpers
# ======================================================================

def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    vx = torch.sqrt((x * x).mean() + eps)
    vy = torch.sqrt((y * y).mean() + eps)
    return float(((x * y).mean() / (vx * vy + eps)).item())


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.argsort(x)
    ranks = torch.empty_like(tmp, dtype=torch.float32)
    ranks[tmp] = torch.arange(len(x), device=x.device, dtype=torch.float32)
    return ranks


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    return pearson_corr(_rankdata(x), _rankdata(y))


@torch.no_grad()
def batch_ic_rankic(
    pred_ret: torch.Tensor,
    y_ret: torch.Tensor,
    ret_mask: torch.Tensor,
) -> Tuple[float, float, int]:
    """
    Compute per-day IC and RankIC, then average.

    Parameters
    ----------
    pred_ret, y_ret, ret_mask : [B, K]
    """
    B = pred_ret.shape[0]
    ics, rics = [], []
    for b in range(B):
        m = ret_mask[b].bool()
        if int(m.sum().item()) < 2:
            continue
        p = pred_ret[b][m].detach().cpu()
        t = y_ret[b][m].detach().cpu()
        ics.append(pearson_corr(p, t))
        rics.append(spearman_corr(p, t))
    if not ics:
        return 0.0, 0.0, 0
    return float(np.mean(ics)), float(np.mean(rics)), len(ics)


# ======================================================================
# Label sanitisation
# ======================================================================

def sanitize_mom_labels(y_mom: torch.Tensor, num_classes: int):
    """
    Normalise classification labels to [0, C-1] with -1 for invalid.

    Returns (y_clean, valid_bool_mask).
    """
    y = y_mom.clone()
    valid = (y >= 0) & (y < num_classes)
    y[~valid] = -1
    return y, valid


# ======================================================================
# LambdaRank NDCG loss (alternative to MSE for return prediction)
# Adapted from reference project 4_train_stage2.py
# ======================================================================

def returns_to_relevance(
    ret: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    n_bins: int = 5,
) -> torch.Tensor:
    """
    Convert continuous returns to discrete relevance levels (0..n_bins-1).
    Uses per-day rank-based binning. Higher return → higher relevance.

    Parameters
    ----------
    ret  : [B, N] future returns
    mask : [B, N] bool, True = valid
    n_bins : number of relevance bins

    Returns
    -------
    rel : [B, N] long tensor, values in 0..n_bins-1
    """
    B, N = ret.shape
    device = ret.device
    if mask is None:
        mask = torch.ones((B, N), dtype=torch.bool, device=device)

    ret_masked = ret.clone()
    ret_masked[~mask] = float("-inf")

    _, indices = torch.sort(ret_masked, dim=1, descending=True)
    ranks = torch.empty_like(indices)
    ranks.scatter_(1, indices, torch.arange(N, device=device).unsqueeze(0).expand(B, N))

    n_valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
    bin_size = n_valid / n_bins
    rel = (n_bins - 1) - torch.floor(ranks.float() / bin_size.expand_as(ranks))
    rel = torch.clamp(rel, min=0, max=n_bins - 1).long()
    rel[~mask] = 0
    return rel


def lambdarank_ndcg_loss(
    scores: torch.Tensor,
    rel: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    k: int = 50,
    sigma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    LambdaRank/LambdaLoss: ΔNDCG@k-weighted pairwise logistic loss.

    Parameters
    ----------
    scores : [B, N] model predicted scores for ranking
    rel    : [B, N] graded relevance (0..n_bins-1), larger = better
    mask   : [B, N] bool, True = valid
    k      : top-k for NDCG
    sigma  : scaling factor for pairwise differences
    """
    B, N = scores.shape
    device = scores.device

    if mask is None:
        mask = torch.ones((B, N), dtype=torch.bool, device=device)

    rel = rel.float()

    # 1) Current ranking → discount (only top-k)
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(N, device=device).expand(B, N))

    discounts = 1.0 / torch.log2(ranks.float() + 2.0)
    discounts = torch.where(ranks < k, discounts, torch.zeros_like(discounts))

    # 2) Gain & IDCG@k
    gains = torch.pow(2.0, rel) - 1.0

    ideal_rel, _ = torch.sort(rel, dim=1, descending=True)
    ideal_gains = torch.pow(2.0, ideal_rel) - 1.0

    pos = torch.arange(N, device=device).float()
    ideal_discounts = 1.0 / torch.log2(pos + 2.0)
    ideal_discounts = torch.where(pos < k, ideal_discounts, torch.zeros_like(ideal_discounts))

    idcg = (ideal_gains * ideal_discounts.unsqueeze(0)).sum(dim=1) + eps

    # 3) Pairwise: only rel_i > rel_j
    s_i = scores.unsqueeze(2)
    s_j = scores.unsqueeze(1)
    score_diff = s_i - s_j

    r_i = rel.unsqueeze(2)
    r_j = rel.unsqueeze(1)

    valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
    pair_mask = ((r_i - r_j) > 0) & valid_pair

    # 4) ΔNDCG weight
    g_i = gains.unsqueeze(2)
    g_j = gains.unsqueeze(1)
    d_i = discounts.unsqueeze(2)
    d_j = discounts.unsqueeze(1)

    delta_dcg = (g_i - g_j).abs() * (d_i - d_j).abs()
    w = torch.clamp(delta_dcg / idcg.view(B, 1, 1), max=10.0)

    # 5) Weighted pairwise logistic loss
    score_diff_clipped = torch.clamp(score_diff, min=-10.0, max=10.0)
    pair_loss = w * F.softplus(-sigma * score_diff_clipped)

    denom = pair_mask.float().sum() + eps
    loss = (pair_loss * pair_mask.float()).sum() / denom

    if not torch.isfinite(loss):
        return torch.zeros((), device=device, requires_grad=True)

    return loss


# ======================================================================
# Loss computation
# ======================================================================

def compute_losses(
    pred_ret: torch.Tensor,
    mom_logits: torch.Tensor,
    y_ret: torch.Tensor,
    y_mom: torch.Tensor,
    ret_mask: torch.Tensor,
    mom_mask: torch.Tensor,
    use_lambdarank: bool = False,
    lambdarank_k: int = 50,
    lambdarank_sigma: float = 1.0,
    lambdarank_bins: int = 5,
):
    """
    Compute regression loss + CE classification loss.

    Regression loss is either MSE or LambdaRank NDCG depending on use_lambdarank.
    Returns (ret_loss, ce_loss, y_valid_mask).
    """
    if use_lambdarank:
        mask_bool = (ret_mask > 0.5)
        rel = returns_to_relevance(y_ret, mask_bool, n_bins=lambdarank_bins)
        ret_loss = lambdarank_ndcg_loss(
            scores=pred_ret, rel=rel, mask=mask_bool,
            k=lambdarank_k, sigma=lambdarank_sigma,
        )
    else:
        diff = pred_ret - y_ret
        mse = (diff * diff) * ret_mask
        ret_loss = mse.sum() / (ret_mask.sum() + 1e-12)

    # Classification: CE with ignore_index=-1
    B, K, C = mom_logits.shape
    y_norm, y_valid = sanitize_mom_labels(y_mom, C)

    logits_flat = mom_logits.reshape(B * K, C)
    y_flat = y_norm.reshape(B * K).long()

    ce_loss = F.cross_entropy(logits_flat, y_flat, reduction="mean", ignore_index=-1)

    return ret_loss, ce_loss, y_valid


# ======================================================================
# Eval metrics
# ======================================================================

@dataclass
class EvalMetrics:
    ret_loss: float = 0.0
    ce_loss: float = 0.0
    total_loss: float = 0.0
    ic: float = 0.0
    rankic: float = 0.0
    cls_acc: float = 0.0
    eval_time_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "ret_loss": self.ret_loss,
            "ce_loss": self.ce_loss,
            "loss": self.total_loss,
            "ic": self.ic,
            "rankic": self.rankic,
            "cls_acc": self.cls_acc,
            "eval_time_s": self.eval_time_s,
        }


# ======================================================================
# Full evaluation loop
# ======================================================================

@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    dl,
    device: torch.device,
    use_amp: bool = False,
    max_batches: Optional[int] = None,
    use_lambdarank: bool = False,
    lambdarank_k: int = 50,
    lambdarank_sigma: float = 1.0,
    lambdarank_bins: int = 5,
) -> EvalMetrics:
    """
    Evaluate LSTM multi-task model on a full dataloader.

    Parameters
    ----------
    model : LSTMMultiTask (or compatible nn.Module).
    dl : DataLoader yielding dicts with X, y_ret, y_mom, ret_mask, mom_mask.
    device : torch device.
    use_amp : Whether to use mixed precision.
    max_batches : Cap the number of batches (for debugging).
    use_lambdarank : Use LambdaRank NDCG loss instead of MSE for eval.

    Returns
    -------
    EvalMetrics with averaged statistics.
    """
    model.eval()

    total_ret_loss = 0.0
    total_ce_loss = 0.0
    total_batches = 0
    total_ic = 0.0
    total_rankic = 0.0
    total_days = 0
    total_acc = 0.0
    total_acc_n = 0

    amp_ctx = torch.amp.autocast(
        device_type="cuda", dtype=torch.float16,
        enabled=(use_amp and device.type == "cuda"),
    )

    t0 = time.time()

    for bi, batch in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break

        X = batch["X"].to(device, non_blocking=True)
        y_ret = batch["y_ret"].to(device, non_blocking=True)
        y_mom = batch["y_mom"].to(device, non_blocking=True)
        ret_mask = batch["ret_mask"].to(device, non_blocking=True)
        mom_mask = batch["mom_mask"].to(device, non_blocking=True)

        B, K, L, D = X.shape
        x_flat = X.reshape(B * K, L, D)

        with amp_ctx:
            pred_ret_flat, mom_logits_flat = model(x_flat)
            pred_ret = pred_ret_flat.reshape(B, K)
            mom_logits = mom_logits_flat.reshape(B, K, -1)

            ret_loss, ce_loss, y_valid = compute_losses(
                pred_ret, mom_logits, y_ret, y_mom, ret_mask, mom_mask,
                use_lambdarank=use_lambdarank,
                lambdarank_k=lambdarank_k,
                lambdarank_sigma=lambdarank_sigma,
                lambdarank_bins=lambdarank_bins,
            )

        total_ret_loss += float(ret_loss.item())
        total_ce_loss += float(ce_loss.item())
        total_batches += 1

        ic, ric, nd = batch_ic_rankic(pred_ret, y_ret, ret_mask)
        total_ic += ic
        total_rankic += ric
        total_days += nd

        # Classification accuracy on valid labels
        pred_cls = mom_logits.argmax(dim=-1)
        y_norm, _ = sanitize_mom_labels(y_mom, mom_logits.shape[-1])
        if int(y_valid.sum().item()) > 0:
            correct = (pred_cls[y_valid] == y_norm[y_valid]).sum().item()
            acc = correct / int(y_valid.sum().item())
            total_acc += float(acc)
            total_acc_n += 1

    elapsed = time.time() - t0
    n = max(total_batches, 1)
    d = max(total_days, 1)

    ret_l = total_ret_loss / n
    ce_l = total_ce_loss / n

    return EvalMetrics(
        ret_loss=ret_l,
        ce_loss=ce_l,
        total_loss=ret_l + ce_l,
        ic=total_ic / d,
        rankic=total_rankic / d,
        cls_acc=total_acc / max(total_acc_n, 1),
        eval_time_s=elapsed,
    )
