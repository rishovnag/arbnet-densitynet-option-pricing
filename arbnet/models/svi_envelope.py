"""SSVI total variance utility.

Implementation of the SSVI (Surface Stochastic Volatility Inspired) parameterization
of total implied variance (Gatheral & Jacquier, 2014). Used as the ground-truth
surface generator in ``arbnet.data.synthetic.SyntheticSurfaceGenerator``.
"""
from __future__ import annotations

import torch


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
