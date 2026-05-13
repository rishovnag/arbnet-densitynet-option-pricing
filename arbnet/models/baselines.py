"""Baseline pricers and competitor models.

Includes:
- BlackScholesPricer: closed-form BS with ATM-fitted sigma per surface snapshot
- HestonPricer: Heston (1993) with characteristic-function pricing via Lewis (2001)
- AckererSoftPenaltyNet: unconstrained MLP with soft no-arbitrage penalties
  (Ackerer, Tagasovska, Vatter, 2020) — the primary competitor we want to beat
  on arbitrage compliance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .composite import bs_call_price, normal_cdf


class BlackScholesPricer(nn.Module):
    """Black-Scholes with a single ATM-fitted vol per snapshot.

    Calibration: choose sigma to minimize MSE on ATM options for each surface.
    Since there is one parameter, this is a degenerate "baseline" giving the floor.
    """

    def __init__(self):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.2)))

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, K, T, S, r, q=None, context=None):
        if q is None:
            q = torch.zeros_like(r)
        sigma_b = self.sigma.expand_as(T)
        price = bs_call_price(S, K, T, r, sigma_b, q)
        w = sigma_b.pow(2) * T
        return {"w": w, "iv": sigma_b, "price": price, "K": K}


class HestonPricer(nn.Module):
    """Heston (1993) stochastic vol pricer via Lewis (2001) Fourier formula.

    Calibration via gradient descent on the price RMSE. Vectorized in the Fourier
    integral using Gaussian-Legendre quadrature.

    Parameters:
        kappa: mean-reversion speed (positive)
        theta: long-run variance (positive)
        sigma: vol of vol (positive)
        rho: spot-vol correlation in (-1, 1)
        v0: initial variance (positive)
    """

    def __init__(self, n_quad: int = 64, u_max: float = 100.0):
        super().__init__()
        # Unconstrained parameterization; transformed via softplus/tanh
        self._kappa = nn.Parameter(torch.tensor(math.log(math.exp(2.0) - 1.0)))
        self._theta = nn.Parameter(torch.tensor(math.log(math.exp(0.04) - 1.0)))
        self._sigma = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self._rho = nn.Parameter(torch.tensor(math.atanh(-0.7)))
        self._v0 = nn.Parameter(torch.tensor(math.log(math.exp(0.04) - 1.0)))
        # Pre-build quadrature nodes / weights on a Gauss-Legendre grid over [eps, u_max]
        nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)
        # Map from [-1, 1] to [eps, u_max]
        eps = 1e-6
        a, b = eps, u_max
        u = 0.5 * (b - a) * nodes_np + 0.5 * (b + a)
        w = 0.5 * (b - a) * weights_np
        self.register_buffer("u_nodes", torch.tensor(u, dtype=torch.float64))
        self.register_buffer("u_weights", torch.tensor(w, dtype=torch.float64))

    @property
    def kappa(self): return F.softplus(self._kappa)
    @property
    def theta(self): return F.softplus(self._theta)
    @property
    def sigma_h(self): return F.softplus(self._sigma)
    @property
    def rho(self): return torch.tanh(self._rho) * 0.999
    @property
    def v0(self): return F.softplus(self._v0)

    def _char_func(self, u: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Heston characteristic function (Lewis form).

        phi(u) = E[exp(i u X_T)] where X_T = log(S_T / F_T) under risk-neutral measure
        (forward-measure simplification: subtract (r - q) drift).
        """
        kappa = self.kappa.to(torch.complex128)
        theta = self.theta.to(torch.complex128)
        sigma = self.sigma_h.to(torch.complex128)
        rho = self.rho.to(torch.complex128)
        v0 = self.v0.to(torch.complex128)
        i = torch.complex(torch.zeros_like(u.real if u.is_complex() else u), torch.ones_like(u.real if u.is_complex() else u)).to(torch.complex128)
        u_c = u.to(torch.complex128)
        T_c = T.to(torch.complex128)
        # Standard Heston char fn (forward log-price)
        xi = kappa - sigma * rho * i * u_c
        d = torch.sqrt(xi**2 + sigma**2 * (u_c**2 + i * u_c))
        g1 = (xi + d) / (xi - d)
        # Use the "little Heston trap" form for numerical stability:
        exp_dT = torch.exp(-d * T_c)
        C = kappa * theta / (sigma**2) * ((xi - d) * T_c - 2.0 * torch.log((1.0 - g1 * exp_dT) / (1.0 - g1)))
        D = ((xi - d) / (sigma**2)) * ((1.0 - exp_dT) / (1.0 - g1 * exp_dT))
        return torch.exp(C + D * v0)

    def forward(self, K, T, S, r, q=None, context=None):
        """Compute call prices via Lewis (2001) integral.

        C(K, T) = S e^{-qT} - sqrt(S K) e^{-(r+q)T/2} / pi * integral_0^inf
                   Re[ e^{i u k} phi(u - i/2) / (u^2 + 1/4) ] du
        where k = log(K / F).
        """
        if q is None:
            q = torch.zeros_like(r)
        F_forward = S * torch.exp((r - q) * T)
        k = torch.log(K / F_forward.clamp(min=1e-12))
        # Move to double precision for stability
        B = T.shape[0]
        u_nodes = self.u_nodes.unsqueeze(0).expand(B, -1)  # (B, Q)
        u_weights = self.u_weights.unsqueeze(0).expand(B, -1)  # (B, Q)
        T_exp = T.unsqueeze(-1).to(torch.float64).expand_as(u_nodes)
        k_exp = k.unsqueeze(-1).to(torch.float64).expand_as(u_nodes)
        # Shift by -i/2:
        u_complex = u_nodes.to(torch.complex128) - 0.5j
        phi = self._char_func(u_complex, T_exp.to(torch.complex128))
        integrand = (torch.exp(1j * u_nodes.to(torch.complex128) * k_exp.to(torch.complex128)) * phi
                     / (u_nodes.to(torch.complex128)**2 + 0.25)).real
        integral = (integrand * u_weights).sum(dim=-1)  # (B,)
        S_d = S.to(torch.float64)
        K_d = K.to(torch.float64)
        r_d = r.to(torch.float64)
        q_d = q.to(torch.float64)
        T_d = T.to(torch.float64)
        price = S_d * torch.exp(-q_d * T_d) - torch.sqrt(S_d * K_d) * torch.exp(-(r_d + q_d) * T_d / 2.0) / math.pi * integral
        price = price.clamp(min=0.0).to(S.dtype)
        w = torch.zeros_like(price)  # placeholder; IV inversion is in iv.py
        iv = torch.zeros_like(price)
        return {"w": w, "iv": iv, "price": price, "K": K.to(S.dtype)}


class AckererSoftPenaltyNet(nn.Module):
    """Unconstrained MLP for total variance with soft no-arbitrage penalties.

    This is the Ackerer, Tagasovska, Vatter (2020) baseline: outputs total
    variance w(k, T; ctx), trained with the price RMSE plus differentiable
    penalties for static no-arbitrage violations.

    The penalty terms are computed and added in losses/pricing.py; this class
    just provides the unconstrained mapping.
    """

    def __init__(self, context_dim: int = 0, hidden_dims=(64, 64, 64)):
        super().__init__()
        in_dim = 2 + context_dim  # (k, T, context)
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, K, T, S, r, q=None, context=None):
        if q is None:
            q = torch.zeros_like(r)
        F_forward = S * torch.exp((r - q) * T)
        k = torch.log(K / F_forward.clamp(min=1e-12))
        k_in = k.unsqueeze(-1)
        T_in = T.unsqueeze(-1)
        if context is not None:
            inp = torch.cat([k_in, T_in, context], dim=-1)
        else:
            inp = torch.cat([k_in, T_in], dim=-1)
        # Softplus + small floor keeps w positive (mild constraint, not full no-arbitrage)
        w = F.softplus(self.net(inp)).squeeze(-1) + 1e-6
        iv = (w / T.clamp(min=1e-6)).sqrt()
        price = bs_call_price(S, K, T, r, iv, q)
        return {"w": w, "iv": iv, "price": price, "K": K}
