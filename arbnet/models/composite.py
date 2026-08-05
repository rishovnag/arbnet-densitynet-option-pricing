"""ArbNet: a direct call-price parametrization free of butterfly and calendar
static arbitrage by construction.

For European calls under constant risk-free rate r and zero dividend yield
(q = 0), the surface

    C(K, T; ctx)  =  e^{-rT} * [ max(S * e^{rT} - K, 0) + Delta(K, T; ctx) ]

is free of butterfly and calendar arbitrage whenever Delta satisfies

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
that the resulting call surface satisfies D1-D4 for every choice of network
weights.

Scope: D1-D4 give exact butterfly (no negative risk-neutral density) and
calendar (carry-adjusted price non-decreasing in T) compliance. They do NOT
cover the Roger Lee wing/tail bound, which is a separate static no-arbitrage
condition; ArbNet's wing slope is bounded only softly by the ICNN's last-layer
norm. Treat tail-bound compliance as monitored, not guaranteed.

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

    # Defaults match the PAPER configuration (J=6, ICNN [64,64], Mono [32,32],
    # ~33k params) -- identical to utils.config.default_arbnet_config, so a
    # bare ArbNet(ArbNetConfig()) is the model the manuscript describes (M4).
    context_dim: int = 0
    n_experts: int = 6
    icnn_hidden: List[int] = field(default_factory=lambda: [64, 64])
    monotone_hidden: List[int] = field(default_factory=lambda: [32, 32])
    activation: str = "softplus"

    # --- Optional localized concave bump (experimental; default OFF) ----------
    # When enabled, Delta gains a non-negative, concave-centered raised-cosine
    # bump at the forward whose *negative curvature mass* is held below the one
    # unit supplied by the intrinsic kink (the (dagger) budget). This is the
    # difference-of-cones / atom-cancellation reformulation. With it ON, A3
    # (expiry), A4 (Delta>=0) and mass-domination remain guaranteed by
    # construction, but POINTWISE butterfly and calendar become *monitored*
    # (not guaranteed) -- a smooth bounded-curvature bump cannot cancel a true
    # Dirac without local violations. Evaluate with the adversarial finer-grid
    # diagnostic (arbnet.arbitrage.adversarial). See bump_prototype.py.
    use_concave_bump: bool = False
    bump_hidden: List[int] = field(default_factory=lambda: [16, 16])
    bump_min_width_frac: float = 0.02  # sigma floor as a fraction of S


# Negative-curvature mass of the unit raised cosine phi(x)=(1+cos(pi x))/2 on
# [-1,1], i.e. int_{-1/2}^{1/2} (-phi'') dx = pi. Used to size the (dagger) gate.
_PHI_NEG_CURV_MASS = math.pi


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

        # Optional concave bump: a small net maps (T, context) -> (m_raw, sigma_raw).
        if config.use_concave_bump:
            dims = [1 + config.context_dim] + list(config.bump_hidden) + [2]
            layers: List[nn.Module] = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(a, b))
                layers.append(nn.Tanh())
            layers = layers[:-1]  # drop trailing activation -> linear head
            self.bump_net = nn.Sequential(*layers)
            # Bias the gate negative so the bump starts ~off (sigmoid(-2)~0.12).
            with torch.no_grad():
                self.bump_net[-1].bias[0] = -2.0
        else:
            self.bump_net = None

    def _delta(
        self,
        K: torch.Tensor,
        T: torch.Tensor,
        S: torch.Tensor,
        context: Optional[torch.Tensor],
        F_T: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Delta(K, T; ctx) >= 0, convex in K, monotone in T.

        With ``use_concave_bump`` enabled, a non-negative concave-centered bump
        at the forward ``F_T`` is added (see :meth:`concave_bump`); it preserves
        Delta>=0 and Delta(.,0)=0 but relaxes pointwise convexity near F_T.
        """
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
        delta = S * boundary * out
        if self.bump_net is not None and F_T is not None:
            bump, _ = self.concave_bump(K, T, F_T, gate_in, boundary)
            delta = delta + bump
        return delta

    def concave_bump(
        self,
        K: torch.Tensor,
        T: torch.Tensor,
        F_T: torch.Tensor,
        gate_in: torch.Tensor,
        boundary: torch.Tensor,
    ):
        """Non-negative concave-centered raised-cosine bump at the forward.

            bump(K,T) = M(T) * phi((K - F_T) / sigma(T)),
            phi(x)    = (1 + cos(pi x))/2  on |x|<=1, else 0  (>=0, concave core)
            M(T)      = (sigma/pi) * (1 - e^{-alpha T}) * sigmoid(m_raw)

        Breeden-Litzenberger: q(K) = delta(K-F_T) + d2Delta/dK2. The bump's core
        has phi'' < 0, so it injects negative curvature that flattens the unit
        intrinsic atom. The (dagger) budget caps the bump's negative-curvature
        mass at  (M/sigma)*pi = (1 - e^{-alpha T}) * sigmoid(m_raw) < 1, the one
        unit the intrinsic supplies -- enforced by construction for every weight.

        The envelope (1 - e^{-alpha T}) makes the bump vanish at T=0 (preserves
        A3); M >= 0 keeps the bump non-negative (preserves A4: Delta >= 0).
        POINTWISE butterfly near F_T and calendar are NOT preserved and must be
        monitored (a smooth bounded-curvature bump cannot cancel a true Dirac).

        Returns (bump, neg_curv_mass) both shape (B,).
        """
        params = self.bump_net(gate_in)                 # (B, 2)
        m_gate = torch.sigmoid(params[:, 0])            # (0,1)
        sigma = (self.cfg.bump_min_width_frac + F.softplus(params[:, 1])) * F_T
        M = (sigma / _PHI_NEG_CURV_MASS) * boundary * m_gate
        x = (K - F_T) / sigma.clamp(min=1e-12)
        phi = torch.where(
            x.abs() <= 1.0,
            0.5 * (1.0 + torch.cos(math.pi * x)),
            torch.zeros_like(x),
        )
        bump = M * phi
        neg_curv_mass = boundary * m_gate               # = (M/sigma)*pi, in [0,1)
        return bump, neg_curv_mass

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
        delta = self._delta(K, T, S, context, F_T=F_T)
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
