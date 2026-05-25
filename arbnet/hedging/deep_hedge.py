"""Deep hedging (Buehler, Gonon, Teichmann, Wood, 2019).

A neural network learns the hedging strategy as a function of state. Used here
both as an evaluation harness (for a frozen pricer, simulate hedging PnL) and
as a training signal for the joint pricing + hedging loss.

We assume a Markovian state and a deterministic policy:
    pi(t, S_t, additional_features) -> hedge position in stock.
The replicating portfolio is rebalanced on a discrete grid; PnL is the
liquidation value at maturity minus the option payoff.
"""
from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn


class DeepHedgeNet(nn.Module):
    """Simple recurrent-MLP policy network for hedging a European call.

    Input at each step: (t_remaining, log(S_t / K), [extra features]).
    Output: stock hedge position (a scalar per path).
    """

    def __init__(self, feature_dim: int = 0, hidden: int = 64):
        super().__init__()
        in_dim = 2 + feature_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t_remaining: torch.Tensor, log_moneyness: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.stack([t_remaining, log_moneyness], dim=-1)
        if extra is not None:
            x = torch.cat([x, extra], dim=-1)
        return self.net(x).squeeze(-1)


def deep_hedge_pnl(
    policy: DeepHedgeNet,
    S_paths: torch.Tensor,
    K: torch.Tensor,
    T_init: torch.Tensor,
    initial_premium: torch.Tensor,
    r: torch.Tensor,
    dt: float,
    q: torch.Tensor = None,
    cost_bps: float = 0.0,
) -> torch.Tensor:
    """Roll the deep hedging policy and return final PnL per path.

    Args:
        policy: DeepHedgeNet (or any callable with same signature).
        S_paths: (n_paths, N+1) spot trajectory.
        K: (n_paths,) strikes.
        T_init: (n_paths,) initial time to expiry.
        initial_premium: (n_paths,) premium received at t=0 (the model's price).
        r, q: rates.
        dt: step size in years.
        cost_bps: trading cost in bps.

    Returns:
        (n_paths,) final PnL.
    """
    n_paths, N1 = S_paths.shape
    N = N1 - 1
    if q is None:
        q = torch.zeros_like(r) if torch.is_tensor(r) else torch.tensor(0.0)
    cost_rate = cost_bps / 1e4

    delta_prev = torch.zeros(n_paths, device=S_paths.device, dtype=S_paths.dtype)
    cash = initial_premium.clone()

    for i in range(N):
        t_remaining = (T_init - i * dt).clamp(min=0.0)
        logm = torch.log((S_paths[:, i] + 1e-12) / (K + 1e-12))
        delta_now = policy(t_remaining, logm)
        trade = delta_now - delta_prev
        # Pay trade cost and stock cost
        cash = cash - trade * S_paths[:, i] - cost_rate * trade.abs() * S_paths[:, i]
        delta_prev = delta_now
        # Cash interest
        if torch.is_tensor(r) and r.dim() > 0:
            cash = cash * (r * dt).exp()
        else:
            cash = cash * math.exp(float(r) * dt)

    # Liquidate at T
    S_T = S_paths[:, -1]
    cash = cash + delta_prev * S_T - cost_rate * delta_prev.abs() * S_T
    payoff = (S_T - K).clamp(min=0.0)
    pnl = cash - payoff
    return pnl
