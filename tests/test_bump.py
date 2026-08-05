"""Tests for the optional localized concave bump (difference-of-cones variant).

These assert the properties that are *by construction* with the bump ON:
  - A3: Delta(.,0) = 0  (price collapses to intrinsic at expiry),
  - A4: Delta >= 0      (price >= discounted forward intrinsic), even with q>0
        and context, for randomly perturbed weights,
  - (dagger) mass budget: the bump's negative-curvature mass stays < 1 unit.

It also checks that the bump is actually active (changes prices near the forward)
and that the adversarial diagnostic reports A4-clean with the bump on, while the
plain (bump-off) ArbNet is fully clean (recovering the by-construction guarantee).

Note: pointwise butterfly/calendar are deliberately NOT asserted with the bump
on -- they are monitored, not guaranteed (see scripts/bump_prototype.py).
"""
import math

import torch
import pytest

from arbnet.models.composite import ArbNet, ArbNetConfig
from arbnet.arbitrage import adversarial_arbitrage_report


def _cfg(context_dim=0):
    return ArbNetConfig(
        context_dim=context_dim, n_experts=4,
        icnn_hidden=[16, 16], monotone_hidden=[16, 16],
        use_concave_bump=True, bump_hidden=[16, 16], bump_min_width_frac=0.02,
    )


def _randomize(model, scale=1.0):
    """Push weights well away from init so guarantees are tested off-init."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * scale)


@pytest.mark.parametrize("context_dim", [0, 3])
def test_expiry_boundary_A3(context_dim):
    torch.manual_seed(0)
    model = ArbNet(_cfg(context_dim))
    _randomize(model)
    B = 64
    K = torch.linspace(15000, 25000, B)
    T = torch.zeros(B)
    S = torch.full((B,), 20000.0)
    r = torch.full((B,), 0.065)
    q = torch.full((B,), 0.012)
    ctx = torch.randn(B, context_dim) if context_dim else None
    out = model(K, T, S, r, q, ctx)
    intrinsic = torch.clamp(S - K, min=0.0)  # at T=0, discounted intrinsic = (S-K)+
    assert torch.allclose(out["price"], intrinsic, atol=1e-4)


@pytest.mark.parametrize("context_dim", [0, 3])
def test_nonnegative_time_value_A4(context_dim):
    """Delta >= 0 for random weights, q>0, and random context."""
    torch.manual_seed(1)
    model = ArbNet(_cfg(context_dim))
    _randomize(model, scale=1.5)
    B = 256
    K = torch.linspace(10000, 30000, B)
    T = torch.rand(B) * 2.0 + 0.01
    S = torch.full((B,), 20000.0)
    r = torch.full((B,), 0.065)
    q = torch.full((B,), 0.012)
    ctx = torch.randn(B, context_dim) if context_dim else None
    out = model(K, T, S, r, q, ctx)
    F_T = S * torch.exp((r - q) * T)
    delta = out["price"] * torch.exp(r * T) - torch.clamp(F_T - K, min=0.0)
    assert delta.min() > -1e-4, f"A4 violated: min Delta = {delta.min().item()}"


def test_mass_budget_dagger():
    """Negative-curvature mass of the bump stays in [0, 1) for any weights."""
    torch.manual_seed(2)
    model = ArbNet(_cfg(context_dim=2))
    for _ in range(5):
        _randomize(model, scale=2.0)
        B = 128
        K = torch.full((B,), 20000.0)
        T = torch.rand(B) * 3.0 + 0.01
        S = torch.full((B,), 20000.0)
        F_T = S * torch.exp(torch.tensor(0.05) * T)
        ctx = torch.randn(B, 2)
        gate_in = torch.cat([T.unsqueeze(-1), ctx], dim=-1)
        alpha = torch.nn.functional.softplus(model.log_alpha)
        boundary = 1.0 - torch.exp(-alpha * T)
        _, neg_mass = model.concave_bump(K, T, F_T, gate_in, boundary)
        assert neg_mass.min() >= 0.0
        assert neg_mass.max() < 1.0


def test_bump_is_active():
    """Enabling the bump must change prices near the forward."""
    torch.manual_seed(3)
    on = ArbNet(_cfg(0))
    _randomize(on)
    # Force a non-trivial gate so the bump is clearly active.
    with torch.no_grad():
        on.bump_net[-1].bias[0] = 2.0
    B = 64
    K = torch.linspace(19000, 21000, B)
    T = torch.full((B,), 0.25)
    S = torch.full((B,), 20000.0)
    r = torch.full((B,), 0.065)
    q = torch.full((B,), 0.0)
    price_on = on(K, T, S, r, q)["price"]
    # Same weights but bump disabled by zeroing the gate mass.
    with torch.no_grad():
        on.bump_net[-1].bias[0] = -50.0  # sigmoid ~ 0 -> bump off
    price_off = on(K, T, S, r, q)["price"]
    assert (price_on - price_off).abs().max() > 1e-3


def test_adversarial_diagnostic_A4_clean_with_bump():
    torch.manual_seed(4)
    model = ArbNet(_cfg(0))
    _randomize(model)
    rep = adversarial_arbitrage_report(model, S=20000.0, r=0.065, q=0.012,
                                       n_base=200, n_refine=200)
    # A4 is guaranteed by construction even with the bump on.
    assert rep.a4_count == 0, str(rep)


def test_plain_arbnet_adversarial_clean():
    """Bump OFF: the adversarial finer-grid check must be fully clean."""
    torch.manual_seed(5)
    cfg = ArbNetConfig(n_experts=4, icnn_hidden=[16, 16], monotone_hidden=[16, 16])
    model = ArbNet(cfg)
    _randomize(model)
    rep = adversarial_arbitrage_report(model, S=20000.0, r=0.065, q=0.012,
                                       n_base=300, n_refine=300, carry_q=False)
    assert rep.butterfly_count == 0, str(rep)
    assert rep.calendar_count == 0, str(rep)
    assert rep.a4_count == 0, str(rep)
