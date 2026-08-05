"""Heston characteristic-function sanity tests (C1 regression guard).

The inverted "little Heston trap" branch (g = (xi+d)/(xi-d)) produced
|phi| ~ 34 and NaN at u -> 0; every price was garbage, silently clamped to 0.
These tests pin the stable Albrecher (2007) branch:

  - phi(0) = 1 and |phi(u)| <= 1 for real u (a valid characteristic function);
  - prices are finite, within the no-arbitrage band
    [(S e^{-qT} - K e^{-rT})+, S e^{-qT}], and non-increasing in strike;
  - the vol-of-vol -> 0 limit recovers Black-Scholes with sigma^2 = v0
    (theta = v0 makes the variance path constant).
"""
import math

import torch
import pytest

from arbnet.models.baselines import HestonPricer
from arbnet.models.composite import bs_call_price


def test_char_func_is_valid():
    model = HestonPricer()
    u = torch.linspace(1e-6, 80.0, 200, dtype=torch.float64)
    T = torch.full_like(u, 0.25)
    phi = model._char_func(u.to(torch.complex128), T.to(torch.complex128))
    assert torch.isfinite(phi.real).all() and torch.isfinite(phi.imag).all()
    mod = phi.abs()
    assert float(mod.max()) <= 1.0 + 1e-8, f"|phi| max {float(mod.max()):.4f} > 1"
    phi0 = model._char_func(torch.tensor([1e-12], dtype=torch.complex128),
                            torch.tensor([0.25], dtype=torch.complex128))
    assert abs(complex(phi0[0]) - 1.0) < 1e-6


def test_prices_in_no_arbitrage_band_and_monotone():
    model = HestonPricer()
    n = 41
    K = torch.linspace(15000.0, 25000.0, n)
    T = torch.full((n,), 0.25)
    S = torch.full((n,), 20000.0)
    r = torch.full((n,), 0.065)
    q = torch.full((n,), 0.012)
    price = model(K, T, S, r, q)["price"]
    assert torch.isfinite(price).all()
    lower = (S * torch.exp(-q * T) - K * torch.exp(-r * T)).clamp(min=0.0)
    upper = S * torch.exp(-q * T)
    assert (price >= lower - 1e-4 * 20000.0).all()
    assert (price <= upper + 1e-6 * 20000.0).all()
    assert (price[1:] - price[:-1] <= 1e-4 * 20000.0).all(), "price not non-increasing in K"


def test_zero_vol_of_vol_limit_recovers_black_scholes():
    model = HestonPricer()
    with torch.no_grad():
        v0 = 0.04
        model._sigma.fill_(math.log(math.exp(1e-4) - 1.0))            # sigma_h ~ 1e-4
        model._theta.fill_(math.log(math.exp(v0) - 1.0))              # theta = v0
        model._v0.fill_(math.log(math.exp(v0) - 1.0))
    n = 21
    K = torch.linspace(17000.0, 23000.0, n)
    T = torch.full((n,), 0.5)
    S = torch.full((n,), 20000.0)
    r = torch.full((n,), 0.065)
    q = torch.full((n,), 0.012)
    heston = model(K, T, S, r, q)["price"]
    bs = bs_call_price(S, K, T, r, torch.full((n,), math.sqrt(v0)), q)
    assert torch.allclose(heston, bs, atol=2.0), \
        f"max |Heston - BS| = {float((heston - bs).abs().max()):.3f} INR"
