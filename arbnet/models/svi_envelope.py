"""SVI / SSVI envelope for tail-bound enforcement.

Implementation of the SVI (Stochastic Volatility Inspired) parameterization
of total implied variance (Gatheral, 2004) and the SSVI surface (Gatheral
& Jacquier, 2014) used as an envelope to enforce Roger Lee (2004) tail bounds
on the asymptotic behavior of the implied vol smile.

The Lee moment formula states that, in absence of arbitrage,
    sigma_IV^2(k, T) * T ~ beta_+/- * |k|   as  k -> +/- infinity,
with 0 <= beta_+/- <= 2. We parameterize the wings via SSVI with constraints
ensuring these bounds hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def ssvi_total_variance(
    k: torch.Tensor,
    theta: torch.Tensor,
    rho: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Compute SSVI total implied variance.

    w(k, T) = (theta / 2) * (1 + rho * phi * k + sqrt((phi * k + rho)^2 + 1 - rho^2))

    where theta = sigma_ATM^2 * T (ATM total variance), rho in (-1, 1) is the
    skew parameter, and phi > 0 controls the smile curvature.

    Args:
        k: Log-moneyness, shape (...,).
        theta: ATM total variance, shape (...,). Must be positive.
        rho: Correlation-like parameter, shape (...,). Must be in (-1, 1).
        phi: Curvature, shape (...,). Must be positive.

    Returns:
        Total implied variance w(k, T), same shape as k.
    """
    inner = (phi * k + rho).pow(2) + (1.0 - rho.pow(2))
    return 0.5 * theta * (1.0 + rho * phi * k + inner.sqrt())


def ssvi_slope_constraint(rho: torch.Tensor, phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Compute the maximum asymptotic slope |beta| of SSVI under given parameters.

    For SSVI, the right wing slope is  beta_+ = (theta * phi * (1 + rho)) / 2  (per unit k)
    in total variance space. For total variance not to violate Roger Lee, we need
    beta_+ <= 2 and beta_- <= 2, i.e. theta * phi * (1 + |rho|) <= 4.

    This function returns max(beta_+, beta_-).
    """
    return 0.5 * theta * phi * (1.0 + rho.abs())


class SVIEnvelope(nn.Module):
    """Neural SSVI envelope.

    A small network maps maturity T (and optionally market context) to SSVI
    parameters (theta, rho, phi). Output parameters are constrained so that
    SSVI tail slopes lie within [0, 2] (Roger Lee bound), giving an automatic
    no-arbitrage envelope for the implied vol smile wings.

    The envelope w_env(k, T) is used as an additive lower bound on total
    variance in the composite model. The body of the surface is fit by an
    ICNN correction (which is positive, so the total stays above the envelope).
    """

    def __init__(self, context_dim: int = 0, hidden_dim: int = 32):
        super().__init__()
        self.context_dim = context_dim
        in_dim = 1 + context_dim  # T plus optional context
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # unbounded (theta_raw, rho_raw, phi_raw)
        )
        # Initialize last-layer biases so the envelope starts flat with sensible ATM level.
        # theta_raw = -3.2  -> softplus ~= 0.04 (typical ATM variance per year)
        # rho_raw   = -0.55 -> tanh ~= -0.5 (typical equity skew)
        # phi_raw   = -8    -> sigmoid ~= 3e-4, so phi is near zero -> essentially flat
        #                       in k. The body of the smile is fit by delta_w; the
        #                       envelope only matters for tail enforcement and slowly
        #                       takes over as training proceeds.
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([-3.2, -0.55, -8.0]))
            self.net[-1].weight.mul_(0.01)

    def parameters_for(self, T: torch.Tensor, context: Optional[torch.Tensor] = None):
        """Return constrained (theta, rho, phi) for each maturity in T."""
        if context is not None:
            inp = torch.cat([T.unsqueeze(-1), context], dim=-1)
        else:
            inp = T.unsqueeze(-1)
        raw = self.net(inp)
        theta_raw, rho_raw, phi_raw = raw[..., 0], raw[..., 1], raw[..., 2]
        # theta > 0; cap so theta * phi * (1+|rho|) <= 4. We construct phi such
        # that this is automatic.
        theta = F.softplus(theta_raw) * T + 1e-6  # ATM total variance grows with T
        rho = torch.tanh(rho_raw) * 0.999  # in (-1, 1) strictly
        # phi <= 4 / (theta * (1 + |rho|)); we set phi = 4 * sigmoid(.) / (theta*(1+|rho|))
        phi_max = 4.0 / (theta * (1.0 + rho.abs()) + 1e-6)
        phi = torch.sigmoid(phi_raw) * phi_max
        return theta, rho, phi

    def forward(self, k: torch.Tensor, T: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute envelope total variance w_env(k, T).

        Args:
            k: log-moneyness, shape (B,).
            T: maturities, shape (B,).
            context: optional context, shape (B, context_dim).

        Returns:
            w_env(k, T), shape (B,).
        """
        theta, rho, phi = self.parameters_for(T, context)
        return ssvi_total_variance(k, theta, rho, phi)
