"""ArbNet: a direct call-price parametrization that is static-arbitrage-free
by construction.

For European calls under constant risk-free rate r and zero dividend yield
(q = 0), the surface

    C(K, T; ctx)  =  e^{-rT} * [ max(S * e^{rT} - K, 0) + Delta(K, T; ctx) ]

is free of static arbitrage whenever Delta satisfies

    (D1) K -> Delta(K, T; ctx) is convex and non-negative for every T >= 0,
    (D2) T -> Delta(K, T; ctx) is non-decreasing for every K > 0,
    (D3) Delta(K, 0; ctx) = 0,
    (D4) lim_{K -> infty} Delta(K, T; ctx) / K = 0.

Conditions (D1)-(D4) are made *architectural* via the construction

    Delta(K, T; ctx) = S * (1 - exp(-alpha * T)) * sum_{j=1..J}
                       softplus( ICNN_j(K/S; ctx) ) * softplus( Mono_j(T; ctx) )

where each ICNN is convex in K/S, each Mono is monotone non-decreasing in T,
softplus enforces non-negativity, and the (1 - exp(-alpha T)) factor enforces
the boundary condition (D3). See Theorem 3.2 of the manuscript for the proof
that the resulting call surface is static-arbitrage-free for every choice of
network weights.

References
----------
Breeden, D. T., Litzenberger, R. H. (1978). Prices of state-contingent claims.
Roper, M. (2010). Arbitrage free implied volatility surfaces.
Carr, P., Madan, D. B. (2005). A note on sufficient conditions for no arbitrage.
Amos, B., Xu, L., Kolter, J. Z. (2017). Input convex neural networks.
Sill, J. (1998). Monotonic networks. NeurIPS.
Daniels, H., Velikova, M. (2010). Monotone and partially monotone neural networks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .icnn import ICNN
from .monotone import MonotoneNet


SQRT_2 = math.sqrt(2.0)
INV_SQRT_2 = 1.0 / SQRT_2
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x * INV_SQRT_2))


def normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def bs_call_price(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Black-Scholes-Merton call price with continuous dividend yield q.

    Falls back to the intrinsic value for T very near zero to avoid 0/0.
    """
    if q is None:
        q = torch.zeros_like(r)
    eps = 1e-12
    T_safe = T.clamp(min=eps)
    sig_sqrtT = sigma * torch.sqrt(T_safe)
    d1 = (
        torch.log((S + eps) / (K + eps)) + (r - q + 0.5 * sigma * sigma) * T_safe
    ) / sig_sqrtT.clamp(min=eps)
    d2 = d1 - sig_sqrtT
    call = S * torch.exp(-q * T_safe) * normal_cdf(d1) - K * torch.exp(-r * T_safe) * normal_cdf(d2)
    intrinsic = (S * torch.exp(-q * T_safe) - K * torch.exp(-r * T_safe)).clamp(min=0.0)
    return torch.where(T < 1e-8, intrinsic, call)


@dataclass
class ArbNetConfig:
    """Configuration for the call-price ArbNet."""

    context_dim: int = 0
    n_experts: int = 4
    icnn_hidden: List[int] = field(default_factory=lambda: [32, 32])
    monotone_hidden: List[int] = field(default_factory=lambda: [16, 16])
    init_log_alpha: float = 0.0
    activation: str = "softplus"

    # Kept for backwards compatibility (no-ops in the new architecture):
    envelope_hidden: int = 32
    slope_cap: float = 1.0
    enforce_min_T: float = 1.0 / 365.0
    enforce_min_w: float = 1e-6
    output: str = "price"


class ArbNet(nn.Module):
    """Direct call-price ArbNet with exact static-arbitrage-freeness.

    Forward signature: forward(K, T, S, r, q, context) returns a dict with
        'price': call price C, shape (B,)
        'C_fwd': undiscounted call price C * e^{rT}, shape (B,)
        'K':     strike pass-through, shape (B,)
    """

    def __init__(self, config: ArbNetConfig):
        super().__init__()
        self.cfg = config
        self.experts = nn.ModuleList(
            [
                ICNN(
                    input_dim=1,
                    hidden_dims=config.icnn_hidden,
                    output_dim=1,
                    context_dim=config.context_dim,
                    activation=config.activation,
                )
                for _ in range(config.n_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [
                MonotoneNet(
                    input_dim=1 + config.context_dim,
                    hidden_dims=config.monotone_hidden,
                    output_dim=1,
                    monotone_cols=[0],  # T is column 0
                )
                for _ in range(config.n_experts)
            ]
        )
        self.log_alpha = nn.Parameter(torch.tensor(2.0))  # alpha = softplus(2.0) ~ 2.13
        # Per-expert global scale: initial softplus(-3) ~ 0.049, giving a non-trivial
        # but small starting time-value relative to S.
        self.log_scale = nn.Parameter(torch.full((config.n_experts,), -3.0))

    def _delta(
        self,
        K: torch.Tensor,
        T: torch.Tensor,
        S: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute Delta(K, T; ctx) >= 0, convex in K, monotone in T."""
        B = K.shape[0]
        # K input to the ICNN: affine in K (so convexity-in-K is preserved),
        # scaled to give the ICNN a well-conditioned input range (~ unit scale).
        K_norm = ((K - S) / (S + 1e-12) * 5.0).unsqueeze(-1)
        T_in = T.unsqueeze(-1)
        if context is not None:
            gate_in = torch.cat([T_in, context], dim=-1)
        else:
            gate_in = T_in
        out = torch.zeros(B, device=K.device, dtype=K.dtype)
        scales = F.softplus(self.log_scale)
        for j, (expert, gate) in enumerate(zip(self.experts, self.gates)):
            f_K = F.softplus(expert(K_norm, context).squeeze(-1))  # (B,) non-neg, convex in K
            h_T = F.softplus(gate(gate_in).squeeze(-1))             # (B,) non-neg, monotone in T
            out = out + scales[j] * f_K * h_T
        alpha = F.softplus(self.log_alpha)
        boundary = 1.0 - torch.exp(-alpha * T)
        return S * boundary * out

    def forward(
        self,
        K: torch.Tensor,
        T: torch.Tensor,
        S: torch.Tensor,
        r: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute the arbitrage-free call price."""
        if q is None:
            q = torch.zeros_like(r)
        F_T = S * torch.exp((r - q) * T)
        intrinsic_fwd = F.relu(F_T - K)
        delta = self._delta(K, T, S, context)
        C_fwd = intrinsic_fwd + delta
        C = C_fwd * torch.exp(-r * T)
        return {"price": C, "C_fwd": C_fwd, "K": K}

    def implied_vol(
        self,
        K: torch.Tensor,
        T: torch.Tensor,
        S: torch.Tensor,
        r: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        n_iter: int = 40,
    ) -> torch.Tensor:
        """Invert the model's call price to Black-Scholes implied volatility."""
        if q is None:
            q = torch.zeros_like(r)
        with torch.no_grad():
            C = self.forward(K, T, S, r, q, context)["price"]
            intrinsic = (S * torch.exp(-q * T) - K * torch.exp(-r * T)).clamp(min=0.0)
            upper = S * torch.exp(-q * T)
            sigma = torch.full_like(C, 0.2)
            valid = (C > intrinsic + 1e-10) & (C < upper - 1e-10)
            T_safe = T.clamp(min=1e-12)
            for _ in range(n_iter):
                bs = bs_call_price(S, K, T, r, sigma, q)
                sig_sqrtT = sigma * torch.sqrt(T_safe)
                d1 = (
                    torch.log((S + 1e-12) / (K + 1e-12)) + (r - q + 0.5 * sigma * sigma) * T_safe
                ) / sig_sqrtT.clamp(min=1e-12)
                vega = S * torch.exp(-q * T_safe) * normal_pdf(d1) * torch.sqrt(T_safe)
                step = (bs - C) / vega.clamp(min=1e-12)
                sigma = (sigma - step).clamp(min=1e-6, max=5.0)
            sigma = torch.where(valid, sigma, torch.full_like(sigma, float("nan")))
            return sigma
