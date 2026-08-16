"""Hedging loss for the joint pricing + hedging training objective."""
from __future__ import annotations

import torch


def hedging_loss(pnl: torch.Tensor, metric: str = "variance", confidence: float = 0.95) -> torch.Tensor:
    """Risk measure on the hedge PnL distribution.

    Args:
        pnl: tensor of shape (n_paths,).
        metric: 'variance', 'cvar', or 'semidev'.
        confidence: confidence level for CVaR (e.g. 0.95).

    Returns:
        Scalar tensor (smaller = better hedger).
    """
    if metric == "variance":
        return pnl.var()
    if metric == "semidev":
        neg = pnl.clamp(max=0.0)
        return neg.pow(2).mean()
    if metric == "cvar":
        # CVaR at level `confidence` for losses (= -pnl).
        losses = -pnl
        k = int((1.0 - confidence) * losses.shape[0])
        k = max(1, k)
        sorted_losses, _ = torch.sort(losses, descending=True)
        return sorted_losses[:k].mean()
    raise ValueError(f"Unknown metric {metric}")
