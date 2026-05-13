"""Batch implied-volatility inverter via Newton's method.

Used for inverting market prices to IVs (input features) and for inverting model
prices to IVs (evaluation). Handles intrinsic-value and upper-bound corner cases.
"""
from __future__ import annotations

import math
import torch

from ..models.composite import bs_call_price, normal_cdf

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _bs_vega(S, K, T, r, sigma, q):
    eps = 1e-12
    sig_sqrtT = (sigma * T.clamp(min=eps).sqrt()).clamp(min=eps)
    d1 = (torch.log((S + eps) / (K + eps)) + (r - q + 0.5 * sigma.pow(2)) * T) / sig_sqrtT
    pdf = (-(d1.pow(2)) * 0.5).exp() / SQRT_2PI
    return S * torch.exp(-q * T) * T.clamp(min=eps).sqrt() * pdf


def implied_vol_newton(
    price: torch.Tensor,
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor = None,
    initial: float = 0.2,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Invert BS call price to implied vol (vectorized).

    Returns NaN where inversion is impossible (price outside no-arbitrage band).
    """
    if q is None:
        q = torch.zeros_like(r)
    intrinsic = (S * torch.exp(-q * T) - K * torch.exp(-r * T)).clamp(min=0.0)
    upper = S * torch.exp(-q * T)
    bad = (price <= intrinsic + 1e-10) | (price >= upper - 1e-10)
    sigma = torch.full_like(price, initial)
    for _ in range(max_iter):
        p = bs_call_price(S, K, T, r, sigma, q)
        vega = _bs_vega(S, K, T, r, sigma, q)
        diff = p - price
        # Newton update with safe vega clamp
        step = diff / vega.clamp(min=1e-10)
        sigma = (sigma - step).clamp(min=1e-6, max=5.0)
        if diff.abs().max().item() < tol:
            break
    sigma = torch.where(bad, torch.full_like(sigma, float("nan")), sigma)
    return sigma
