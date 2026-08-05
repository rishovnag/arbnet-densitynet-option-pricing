"""Property tests for DensityNet -- the paper's MAIN theorem (H6 closure).

Asserts, for RANDOM parameter vectors (not just the init) and random context:

  - martingale normalisation: sum_i w_i m_i = 1 exactly (mixture mean 1);
  - A1 butterfly, A2 calendar (fixed FORWARD-MONEYNESS -- the exact model-free
    condition Theorem (density) (iii) guarantees), and A4 on the adversarial
    finer, forward-refined float64 grid, across q in {0, 1.2%, 2%};
  - the calendar convex-order property directly on nc(kappa, T);
  - the expiry boundary: exact intrinsic AT T = 0, and nc(kappa, 0+) >=
    (1 - kappa)+ as T -> 0 (the surface sits ABOVE intrinsic in the limit --
    the documented boundary approximation, not below it).

Also the plain-ArbNet analogue (H6): butterfly, A4 and the fixed-strike
e^{rT}C calendar (the quantity Theorem 4.1 guarantees) must be exactly clean
for random weights x context x q. NOTE: ArbNet's fixed-forward-moneyness
calendar count is deliberately NOT asserted -- it is monitored disclosure,
not a guarantee (H5).
"""
import torch
import pytest

from arbnet.models.density import DensityNet, DensityNetConfig
from arbnet.models.composite import ArbNet, ArbNetConfig
from arbnet.arbitrage import adversarial_arbitrage_report

S0, R = 20000.0, 0.065


def _randomize(model, scale=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)


def _density(context_dim=0, seed=0, scale=0.5):
    torch.manual_seed(seed)
    m = DensityNet(DensityNetConfig(context_dim=context_dim, n_components=8))
    _randomize(m, scale=scale, seed=seed)
    return m


# ---------------------------------------------------------------------------
# Martingale normalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("context_dim", [0, 10])
def test_martingale_normalisation(context_dim):
    model = _density(context_dim, seed=1)
    B = 16
    ctx = torch.randn(B, context_dim) if context_dim else None
    w, m, c0, c1 = model._params(ctx, B, torch.device("cpu"), torch.float32)
    assert torch.allclose((w * m).sum(-1), torch.ones(B), atol=1e-5)
    assert (w >= 0).all() and (m > 0).all() and (c0 >= 0).all() and (c1 >= 0).all()


# ---------------------------------------------------------------------------
# Adversarial compliance for random theta (the theorem, empirically)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", [0.0, 0.012, 0.02])
@pytest.mark.parametrize("context_dim", [0, 10])
def test_density_adversarial_clean_random_theta(q, context_dim):
    for seed in range(3):
        model = _density(context_dim, seed=seed)
        ctx = torch.randn(1, context_dim, generator=torch.Generator().manual_seed(seed)) \
            if context_dim else None
        rep = adversarial_arbitrage_report(
            model, S=S0, r=R, q=q, context=ctx,
            n_base=200, n_refine=200,          # smaller grid: test speed
        )
        assert rep.butterfly_count == 0, f"seed {seed}: {rep}"
        assert rep.calendar_count == 0, f"seed {seed}: {rep}"   # fixed fwd-moneyness (exact)
        assert rep.a4_count == 0, f"seed {seed}: {rep}"


# ---------------------------------------------------------------------------
# Calendar: convex order directly on nc(kappa, T)
# ---------------------------------------------------------------------------
def test_density_nc_monotone_in_T_at_fixed_kappa():
    model = _density(0, seed=3).double()
    kappa = torch.linspace(0.6, 1.4, 81, dtype=torch.float64)
    T_grid = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    prev = None
    for T in T_grid:
        F_T = S0 * torch.exp(torch.tensor((R - 0.012) * T, dtype=torch.float64))
        K = kappa * F_T
        n = K.shape[0]
        out = model(K, torch.full((n,), T, dtype=torch.float64),
                    torch.full((n,), S0, dtype=torch.float64),
                    torch.full((n,), R, dtype=torch.float64),
                    torch.full((n,), 0.012, dtype=torch.float64), None)
        nc = out["price"] * torch.exp(torch.tensor(R * T, dtype=torch.float64)) / F_T
        if prev is not None:
            assert (nc - prev >= -1e-10).all(), f"nc decreased between maturities at T={T}"
        prev = nc


# ---------------------------------------------------------------------------
# Expiry boundary: exact at T=0; ABOVE intrinsic (never below) as T -> 0+
# ---------------------------------------------------------------------------
def test_density_expiry_boundary():
    model = _density(0, seed=4)
    B = 64
    K = torch.linspace(15000.0, 25000.0, B)
    S = torch.full((B,), S0)
    r = torch.full((B,), R)
    q = torch.full((B,), 0.012)
    # exact intrinsic at T = 0 (the boundary convention)
    out0 = model(K, torch.zeros(B), S, r, q, None)
    assert torch.allclose(out0["price"], (S - K).clamp(min=0.0), atol=1e-4)
    # as T -> 0+, the surface collapses to a residual smile ABOVE intrinsic
    for T in (1e-4, 1e-3, 1e-2):
        outT = model(K, torch.full((B,), T), S, r, q, None)
        intrinsic = torch.exp(-r * T) * (S * torch.exp((r - q) * T) - K).clamp(min=0.0)
        assert (outT["price"] - intrinsic >= -1e-6 * S0).all()


# ---------------------------------------------------------------------------
# H6: plain ArbNet -- the GUARANTEED conditions, random weights x context x q
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", [0.0, 0.012, 0.02])
@pytest.mark.parametrize("context_dim", [0, 10])
def test_arbnet_guaranteed_conditions_random_theta(q, context_dim):
    for seed in range(2):
        torch.manual_seed(seed)
        model = ArbNet(ArbNetConfig(context_dim=context_dim, n_experts=4,
                                    icnn_hidden=[16, 16], monotone_hidden=[16, 16]))
        _randomize(model, scale=0.5, seed=seed)
        ctx = torch.randn(1, context_dim, generator=torch.Generator().manual_seed(seed)) \
            if context_dim else None
        rep = adversarial_arbitrage_report(
            model, S=S0, r=R, q=q, context=ctx,
            n_base=200, n_refine=200, carry_q=False,   # e^{rT}C: the guaranteed carry
        )
        assert rep.butterfly_count == 0, f"seed {seed}: {rep}"
        assert rep.calendar_strike_count == 0, f"seed {seed}: {rep}"  # Theorem 4.1 quantity
        assert rep.a4_count == 0, f"seed {seed}: {rep}"
        # rep.calendar_count (fixed fwd-moneyness) is MONITORED for ArbNet, not
        # guaranteed (H5) -- intentionally not asserted.
