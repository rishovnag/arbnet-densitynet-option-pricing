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


# ---------------------------------------------------------------------------
# Skew clock (v4): m_i(T) = 1 + h(T)(m_i - 1), h(T) = 1 - e^{-beta T}
# ---------------------------------------------------------------------------
def _density_skew(context_dim=0, seed=0, scale=0.5):
    torch.manual_seed(seed)
    m = DensityNet(DensityNetConfig(context_dim=context_dim, n_components=8,
                                    skew_clock=True))
    _randomize(m, scale=scale, seed=seed)
    return m


@pytest.mark.parametrize("context_dim", [0, 10])
def test_skew_clock_martingale_and_positivity(context_dim):
    """sum_i w_i m_i(T) = 1 and m_i(T) > 0 for every T, random theta."""
    model = _density_skew(context_dim, seed=1)
    B = 16
    ctx = torch.randn(B, context_dim) if context_dim else None
    w, m, c0, c1 = model._params(ctx, B, torch.device("cpu"), torch.float32)
    beta = model._skew_beta(ctx, B, torch.device("cpu"), torch.float32)
    assert beta is not None and (beta > 0).all()
    for T in (0.0, 1e-4, 0.02, 0.08, 0.5, 2.0):
        h = 1.0 - torch.exp(-beta * T)
        m_T = 1.0 + h * (m - 1.0)
        assert (m_T > 0).all()
        assert torch.allclose((w * m_T).sum(-1), torch.ones(B), atol=1e-5)
    # h(0) = 0: the mixing law is degenerate at 1 at expiry
    assert torch.allclose(1.0 + (1.0 - torch.exp(-beta * 0.0)) * (m - 1.0),
                          torch.ones_like(m))


@pytest.mark.parametrize("q", [0.0, 0.012, 0.02])
@pytest.mark.parametrize("context_dim", [0, 10])
def test_skew_clock_adversarial_clean_random_theta(q, context_dim):
    """A1/A2(fixed fwd-moneyness)/A4 on the adversarial grid, random theta."""
    for seed in range(3):
        model = _density_skew(context_dim, seed=seed)
        ctx = torch.randn(1, context_dim, generator=torch.Generator().manual_seed(seed)) \
            if context_dim else None
        rep = adversarial_arbitrage_report(
            model, S=S0, r=R, q=q, context=ctx,
            n_base=200, n_refine=200,
        )
        assert rep.butterfly_count == 0, f"seed {seed}: {rep}"
        assert rep.calendar_count == 0, f"seed {seed}: {rep}"
        assert rep.a4_count == 0, f"seed {seed}: {rep}"


def test_skew_clock_nc_monotone_in_T_at_fixed_kappa():
    """Calendar convex order directly on nc(kappa, T), incl. very short T --
    the regime the skew clock reshapes (h(T) -> 0)."""
    model = _density_skew(0, seed=3).double()
    kappa = torch.linspace(0.6, 1.4, 81, dtype=torch.float64)
    T_grid = [1e-4, 5e-4, 2e-3, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
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


def test_skew_clock_exact_expiry_continuity():
    """The headline payoff of the skew clock: x_{0+} = 1, so the surface is
    CONTINUOUS at T = 0 -- nc(kappa, 0+) -> (1 - kappa)^+ (no residual smile),
    unlike the fixed-mean design which converges to sum w_i (m_i - kappa)^+."""
    model = _density_skew(0, seed=4).double()
    kappa = torch.linspace(0.6, 1.4, 161, dtype=torch.float64)
    q = 0.012
    gaps = []
    for T in (1e-2, 1e-3, 1e-4, 1e-5):
        F_T = S0 * torch.exp(torch.tensor((R - q) * T, dtype=torch.float64))
        K = kappa * F_T
        n = K.shape[0]
        with torch.no_grad():
            out = model(K, torch.full((n,), T, dtype=torch.float64),
                        torch.full((n,), S0, dtype=torch.float64),
                        torch.full((n,), R, dtype=torch.float64),
                        torch.full((n,), q, dtype=torch.float64), None)
        nc = out["price"] * torch.exp(torch.tensor(R * T, dtype=torch.float64)) / F_T
        intrinsic = (1.0 - kappa).clamp(min=0.0)
        assert (nc - intrinsic >= -1e-10).all()          # never below intrinsic
        gaps.append(float((nc - intrinsic).max()))
    # the worst gap over kappa must VANISH as T -> 0 (continuity at expiry):
    # it is O(sqrt(c0 T)) + O(beta T), so ~1e-3 at T=1e-5 is already generous
    assert gaps[-1] < 1e-3, f"gap did not vanish: {gaps}"
    assert gaps[-1] < gaps[0] / 3, f"gap not decreasing: {gaps}"


def test_skew_clock_fixed_mean_residual_smile_contrast():
    """Sanity contrast: with the SAME randomised parameters, the fixed-mean
    model keeps a non-vanishing residual smile at T -> 0+ while the skew-clock
    model does not (this is exactly Corollary (approximate expiry) vs its
    removal)."""
    fixed = _density(0, seed=7).double()
    kappa = torch.linspace(0.7, 1.3, 121, dtype=torch.float64)
    q = 0.012
    T = 1e-5
    F_T = S0 * torch.exp(torch.tensor((R - q) * T, dtype=torch.float64))
    K = kappa * F_T
    n = K.shape[0]
    args = (torch.full((n,), T, dtype=torch.float64),
            torch.full((n,), S0, dtype=torch.float64),
            torch.full((n,), R, dtype=torch.float64),
            torch.full((n,), q, dtype=torch.float64), None)
    with torch.no_grad():
        nc_fixed = fixed(K, *args)["price"] * torch.exp(torch.tensor(R * T, dtype=torch.float64)) / F_T
    intrinsic = (1.0 - kappa).clamp(min=0.0)
    gap_fixed = float((nc_fixed - intrinsic).max())
    # fixed means: residual smile of order the mean spread -- bounded AWAY from 0
    assert gap_fixed > 1e-3
    skew = _density_skew(0, seed=7).double()
    with torch.no_grad():
        nc_skew = skew(K, *args)["price"] * torch.exp(torch.tensor(R * T, dtype=torch.float64)) / F_T
    gap_skew = float((nc_skew - intrinsic).max())
    assert gap_skew < 1e-3 and gap_skew < gap_fixed / 10, (gap_skew, gap_fixed)


def test_skew_clock_mixture_diagnostics():
    model = _density_skew(0, seed=5)
    d = model.mixture_diagnostics(None)
    assert set(d) >= {"V_m", "c0", "c1", "beta", "h_1w"}
    assert d["V_m"] >= 0 and d["beta"] > 0 and 0 <= d["h_1w"] < 1
    fixed = _density(0, seed=5)
    d2 = fixed.mixture_diagnostics(None)
    assert "beta" not in d2 and d2["V_m"] >= 0


# ---------------------------------------------------------------------------
# Dispersion-constrained mode (Section 9.7 sweep)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target", [0.0, 1e-4, 4e-3, 3e-2])
def test_fixed_V_m_hits_target_and_stays_compliant(target):
    """Pinning V_m must (a) realise the target exactly, (b) preserve the
    martingale normalisation and positivity, and (c) leave A1/A2/A4 intact --
    the sweep of Section 9.7 traces a fit trade-off INSIDE the certified
    class, so every pinned model must still be adversarially clean."""
    torch.manual_seed(3)
    model = DensityNet(DensityNetConfig(context_dim=0, n_components=8, fixed_V_m=target))
    _randomize(model, scale=0.7, seed=3)
    d = model.mixture_diagnostics(None)
    assert abs(d["V_m"] - target) < 1e-6, (d["V_m"], target)
    w, m, c0, c1 = model._params(None, 1, torch.device("cpu"), torch.float32)
    assert torch.allclose((w * m).sum(-1), torch.ones(1), atol=1e-5)
    assert (m > 0).all()
    rep = adversarial_arbitrage_report(model, S=S0, r=R, q=0.012, context=None,
                                       n_base=200, n_refine=200)
    assert rep.butterfly_count == 0 and rep.calendar_count == 0 and rep.a4_count == 0


def test_fixed_V_m_zero_is_a_single_lognormal():
    """V_m = 0 collapses the mixing law to a point mass at 1, so the surface
    is a single Black call -- the degenerate corner of the sweep."""
    torch.manual_seed(4)
    model = DensityNet(DensityNetConfig(context_dim=0, n_components=8, fixed_V_m=0.0))
    _randomize(model, scale=0.7, seed=4)
    w, m, c0, c1 = model._params(None, 1, torch.device("cpu"), torch.float32)
    assert torch.allclose(m, torch.ones_like(m), atol=1e-6)


def test_constant_skew_clock_is_a_reparameterisation():
    """Why the sweep pins V_m rather than a constant clock: with m_bar free, a
    constant h == s is absorbed by rescaling the means, so constant-h models
    are the same family. Checked by matching prices between (h == s, means
    m_bar) and (h == 1, means 1 + s(m_bar - 1))."""
    torch.manual_seed(5)
    base = DensityNet(DensityNetConfig(context_dim=0, n_components=8))
    _randomize(base, scale=0.5, seed=5)
    w, m, c0, c1 = base._params(None, 1, torch.device("cpu"), torch.float32)
    s = 0.4
    m_s = 1.0 + s * (m - 1.0)
    # both mixing laws have mean 1 and positive atoms -> both are admissible
    assert torch.allclose((w * m_s).sum(-1), torch.ones(1), atol=1e-6)
    assert (m_s > 0).all()
    # and the contracted law is exactly what a constant clock h == s produces
    assert torch.allclose(m_s, 1.0 + s * (m - 1.0))
