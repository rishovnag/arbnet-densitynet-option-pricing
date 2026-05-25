"""Classical Black-Scholes delta hedging utilities.

Used as a baseline against deep hedging. Given a pricer that produces implied
vols, we can compute model-implied BS deltas and simulate discrete hedging on
realized paths.
"""
from __future__ import annotations

import math
import torch

from ..models.composite import normal_cdf


def bs_delta(S: torch.Tensor, K: torch.Tensor, T: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor, q: torch.Tensor = None) -> torch.Tensor:
    """BS call delta with continuous dividend yield."""
    if q is None:
        q = torch.zeros_like(r)
    eps = 1e-12
    sig_sqrtT = (sigma * (T.clamp(min=eps).sqrt())).clamp(min=eps)
    d1 = (torch.log((S + eps) / (K + eps)) + (r - q + 0.5 * sigma.pow(2)) * T) / sig_sqrtT
    return torch.exp(-q * T) * normal_cdf(d1)


def hedge_pnl_delta(
    S_paths: torch.Tensor,
    K: torch.Tensor,
    T_init: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    dt: float,
    q: torch.Tensor = None,
    cost_bps: float = 0.0,
) -> torch.Tensor:
    """Simulate discrete delta hedging PnL of a short call position.

    Args:
        S_paths: shape (n_paths, N+1), spot trajectory.
        K: strike (scalar or shape (n_paths,)).
        T_init: time to expiry at t=0, shape (n_paths,) or scalar.
        r, q: rates, scalar or per-path.
        sigma: implied vol used for delta (scalar or per-path).
        dt: step size in years.
        cost_bps: per-trade transaction cost in basis points of notional.

    Returns:
        Final hedged PnL per path, shape (n_paths,). Positive = profit for the
        hedger; the metric of interest is the standard deviation of this quantity.
    """
    n_paths, N1 = S_paths.shape
    N = N1 - 1
    if q is None:
        q = torch.zeros_like(r) if torch.is_tensor(r) else torch.tensor(0.0)
    if not torch.is_tensor(K):
        K = torch.tensor(K, dtype=S_paths.dtype, device=S_paths.device)
    if K.dim() == 0:
        K = K.expand(n_paths)
    if not torch.is_tensor(T_init):
        T_init = torch.tensor(T_init, dtype=S_paths.dtype, device=S_paths.device)
    if T_init.dim() == 0:
        T_init = T_init.expand(n_paths)
    if torch.is_tensor(r) and r.dim() == 0:
        r = r.expand(n_paths)
    if not torch.is_tensor(r):
        r = torch.full((n_paths,), float(r), dtype=S_paths.dtype, device=S_paths.device)
    if torch.is_tensor(q) and q.dim() == 0:
        q = q.expand(n_paths)
    if not torch.is_tensor(q):
        q = torch.full((n_paths,), float(q), dtype=S_paths.dtype, device=S_paths.device)
    if not torch.is_tensor(sigma):
        sigma = torch.full((n_paths,), float(sigma), dtype=S_paths.dtype, device=S_paths.device)
    if sigma.dim() == 0:
        sigma = sigma.expand(n_paths)
    cost_rate = cost_bps / 1e4

    # Initial: short one call, receive premium C_0, set hedge delta_0 in stock
    T0 = T_init
    delta_prev = bs_delta(S_paths[:, 0], K, T0, r, sigma, q)
    # Cash from sale of call:
    from ..models.composite import bs_call_price
    C0 = bs_call_price(S_paths[:, 0], K, T0, r, sigma, q)
    cash = C0 - delta_prev * S_paths[:, 0]
    cash = cash - cost_rate * delta_prev.abs() * S_paths[:, 0]

    for i in range(1, N):
        T_now = (T0 - i * dt).clamp(min=1e-6)
        delta_now = bs_delta(S_paths[:, i], K, T_now, r, sigma, q)
        # Rebalance: trade (delta_now - delta_prev) shares at S_paths[:, i]
        trade = delta_now - delta_prev
        cash = cash * math.exp(r.mean().item() * dt) if r.dim() == 0 else cash * (r * dt).exp()
        # (Per-path interest accrual is approximated via mean rate for vectorization;
        # for production use, expand to per-path exp).
        cash = cash - trade * S_paths[:, i] - cost_rate * trade.abs() * S_paths[:, i]
        delta_prev = delta_now

    # Settle: payoff of short call = -max(S_T - K, 0); plus stock holding sold at S_T
    S_T = S_paths[:, -1]
    cash = cash * (r * dt).exp() if r.dim() == 1 else cash * math.exp(r.mean().item() * dt)
    cash = cash + delta_prev * S_T - cost_rate * delta_prev.abs() * S_T
    payoff = (S_T - K).clamp(min=0.0)
    pnl = cash - payoff
    return pnl
