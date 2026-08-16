"""Pricing losses.

Includes:
- price_rmse / iv_rmse: standard fit losses
- soft_no_arbitrage_penalty: differentiable penalty (Ackerer et al. 2020) for
  butterfly, calendar, and tail violations. Used by AckererSoftPenaltyNet.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def price_rmse(pred: torch.Tensor, target: torch.Tensor, vega: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Vega-weighted RMSE on prices when vega provided, else raw RMSE."""
    diff = pred - target
    if vega is not None:
        diff = diff / vega.clamp(min=1e-4)
    return diff.pow(2).mean().sqrt()


def iv_rmse(pred_iv: torch.Tensor, target_iv: torch.Tensor) -> torch.Tensor:
    return (pred_iv - target_iv).pow(2).mean().sqrt()


def soft_no_arbitrage_penalty(
    model: torch.nn.Module,
    K_grid: torch.Tensor,
    T_grid: torch.Tensor,
    S: torch.Tensor,
    r: torch.Tensor,
    q: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    tail_bound: float = 2.0,
) -> dict:
    """Differentiable soft no-arbitrage penalty on a (K, T) grid.

    Computes three penalty terms (Ackerer et al. 2020):
        1. butterfly: relu(-d^2 C / d K^2) integrated over the grid.
        2. calendar:  relu(-d (e^{rT} C) / d T) integrated.
        3. tail:      relu(slope_tail - bound).

    Implemented via finite differences on the grid (cheap and differentiable).

    Args:
        model: any nn.Module with forward(K, T, S, r, q, context) -> dict {'price'}.
        K_grid: 1D grid of strikes, shape (n_K,).
        T_grid: 1D grid of maturities, shape (n_T,).
        S, r, q: scalars or shape (1,).

    Returns:
        Dict with 'butterfly', 'calendar', 'tail', 'total'.
    """
    if q is None:
        q = torch.zeros_like(r)
    n_K = K_grid.shape[0]
    n_T = T_grid.shape[0]
    T_mesh, K_mesh = torch.meshgrid(T_grid, K_grid, indexing="ij")  # (n_T, n_K)
    T_flat = T_mesh.flatten()
    K_flat = K_mesh.flatten()
    B = T_flat.shape[0]
    device = K_grid.device
    dtype = K_grid.dtype
    S_b = S.expand(B) if S.dim() > 0 else torch.full((B,), float(S), device=device, dtype=dtype)
    r_b = r.expand(B) if r.dim() > 0 else torch.full((B,), float(r), device=device, dtype=dtype)
    q_b = q.expand(B) if q.dim() > 0 else torch.full((B,), float(q), device=device, dtype=dtype)
    ctx_b = context.unsqueeze(0).expand(B, -1) if context is not None else None
    out = model(K_flat, T_flat, S_b, r_b, q_b, ctx_b)
    C = out["price"].view(n_T, n_K)
    # Carry-adjusted price = e^{rT} * C, used for the calendar check
    T_grid_col = T_grid.view(n_T, 1)
    r_val = r_b[0]
    C_fwd = C * torch.exp(r_val * T_grid_col)

    # --- Butterfly: second difference of C wrt K (per T row)
    K_l = K_grid[:-2]
    K_m = K_grid[1:-1]
    K_r = K_grid[2:]
    C_l = C[:, :-2]
    C_m = C[:, 1:-1]
    C_r = C[:, 2:]
    h1 = (K_m - K_l).clamp(min=1e-8)
    h2 = (K_r - K_m).clamp(min=1e-8)
    slope_lo = (C_m - C_l) / h1
    slope_hi = (C_r - C_m) / h2
    d2 = (slope_hi - slope_lo) / ((h1 + h2) / 2.0)
    butterfly = F.relu(-d2).mean()

    # --- Calendar: difference of e^{rT} C wrt T (per K column)
    fwd_dT = C_fwd[1:, :] - C_fwd[:-1, :]
    calendar = F.relu(-fwd_dT).mean()

    # --- Tail: estimate left/right wing slopes of total variance via OLS on 20% tails
    # We use the implied vol -> w mapping if available, else approximate via log-price.
    if "w" in out:
        w = out["w"].view(n_T, n_K)
        # Approximate log-moneyness from strikes
        log_K = torch.log(K_grid.clamp(min=1e-12))
        tail_frac = 0.2
        n_tail = max(2, int(n_K * tail_frac))
        k_right = log_K[-n_tail:]
        w_right = w[:, -n_tail:]
        k_left = log_K[:n_tail]
        w_left = w[:, :n_tail]
        def _slope(x, y):
            xm = x.mean()
            ym = y.mean(dim=-1, keepdim=True)
            num = ((x - xm) * (y - ym)).sum(dim=-1)
            den = ((x - xm) ** 2).sum().clamp(min=1e-8)
            return num / den
        slope_r = _slope(k_right, w_right)
        slope_l = -_slope(k_left, w_left)
        tail = F.relu(slope_r - tail_bound).mean() + F.relu(slope_l - tail_bound).mean()
    else:
        tail = torch.zeros((), device=device, dtype=dtype)

    total = butterfly + calendar + tail
    return {"butterfly": butterfly, "calendar": calendar, "tail": tail, "total": total}
