"""Adversarial static-arbitrage diagnostic for the concave-bump ArbNet.

Once the localized concave bump is enabled (``ArbNetConfig.use_concave_bump``),
pointwise butterfly and calendar compliance are no longer guaranteed by
construction -- only A3 (expiry), A4 (Delta>=0) and the (dagger) negative-curvature
mass budget are. A smooth, bounded-curvature bump cannot cancel a true Dirac
atom without creating local convexity violations *between* the training grid
points, right beside the forward (see scripts/bump_prototype.py). A diagnostic
sampled on the training grid would therefore report a *false zero*.

This module re-checks a fitted model on a grid that is

  1. finer than the training grid, and
  2. specifically refined in a cluster around K = F_T for every maturity,

and additionally verifies A4 (level domination, Delta >= 0), which the single
cone made automatic but the bump does not. All differencing is done in float64.

It works for any model exposing ``forward(K, T, S, r, q, context) -> {'price'}``,
so it can also be run against the plain (bump-off) ArbNet as a sanity check
(there it must report exactly zero, recovering the by-construction guarantee).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AdversarialArbReport:
    n_T: int
    n_K: int
    butterfly_count: int
    calendar_count: int        # calendar at fixed FORWARD-MONEYNESS (standard condition)
    a4_count: int
    butterfly_worst: float
    calendar_worst: float
    a4_worst: float
    worst_butterfly_K: float
    worst_butterfly_T: float
    tol: float
    calendar_strike_count: int = 0   # calendar at fixed STRIKE (reference; e^{rT}C nondecr in T)
    calendar_strike_worst: float = 0.0

    @property
    def clean(self) -> bool:
        return self.butterfly_count == 0 and self.calendar_count == 0 and self.a4_count == 0

    def __str__(self) -> str:
        return (
            f"Adversarial arb report on {self.n_T}x{self.n_K} float64 grid "
            f"(tol={self.tol:.1e}): "
            f"butterfly {self.butterfly_count} (worst {self.butterfly_worst:.3e} "
            f"at K={self.worst_butterfly_K:.1f}, T={self.worst_butterfly_T:.4f}), "
            f"calendar[fwd-moneyness] {self.calendar_count} (worst {self.calendar_worst:.3e}), "
            f"calendar[strike] {self.calendar_strike_count} (worst {self.calendar_strike_worst:.3e}), "
            f"A4 {self.a4_count} (worst {self.a4_worst:.3e}) -> "
            f"{'CLEAN' if self.clean else 'VIOLATIONS'}"
        )


def _dedupe_sorted(x: torch.Tensor, min_gap: float) -> torch.Tensor:
    """Drop points closer than ``min_gap`` to their kept predecessor.

    ``torch.unique`` on the merged base+refine grids can leave *near*-duplicate
    neighbours (e.g. a base strike 1e-9 away from a refine strike). Divided
    second differences with such a tiny h amplify float64 round-off into
    spurious O(1e-3) "butterfly violations" on surfaces that are analytically
    convex (this produced 48 false-positive days for DensityNet in
    nse_density_v2.json). Enforcing a minimum spacing keeps the differencing
    noise ~1e-7, comfortably below the 1e-6 tolerance.
    """
    if x.numel() <= 1:
        return x
    keep = torch.ones_like(x, dtype=torch.bool)
    last = float(x[0])
    for i in range(1, x.numel()):
        if float(x[i]) - last > min_gap:
            last = float(x[i])
        else:
            keep[i] = False
    return x[keep]


def _moneyness_grid(span: float, refine_span: float, n_base: int, n_refine: int) -> torch.Tensor:
    """Forward-moneyness grid kappa = K/F_T around 1, with a dense cluster at the forward."""
    base = torch.exp(torch.linspace(-span, span, n_base, dtype=torch.float64))
    refine = torch.linspace(1.0 - refine_span, 1.0 + refine_span, n_refine, dtype=torch.float64)
    kappa = torch.unique(torch.cat([base, refine]), sorted=True)
    kappa = kappa[kappa > 0]
    return _dedupe_sorted(kappa, min_gap=1e-6)


def _refined_strike_grid(
    S: float, r: float, q: float, T_grid: torch.Tensor,
    n_base: int, n_refine: int, span: float, refine_span: float,
) -> torch.Tensor:
    """Common strike grid: a wide base linspace plus dense clusters at each F_T."""
    lo, hi = S * (1.0 - span), S * (1.0 + span)
    pts = [torch.linspace(lo, hi, n_base, dtype=torch.float64)]
    for T in T_grid.tolist():
        F_T = S * float(torch.exp(torch.tensor((r - q) * T)))
        w = refine_span * F_T
        pts.append(torch.linspace(F_T - w, F_T + w, n_refine, dtype=torch.float64))
    K = torch.cat(pts)
    K = torch.unique(K, sorted=True)
    K = K[(K > 0)]
    return _dedupe_sorted(K, min_gap=1e-5 * S)   # ~0.2 INR at Nifty scale


@torch.no_grad()
def adversarial_arbitrage_report(
    model: torch.nn.Module,
    S: float,
    r: float,
    q: float = 0.0,
    T_grid: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    n_base: int = 600,
    n_refine: int = 400,
    span: float = 0.6,
    refine_span: float = 0.05,
    tol: float = 1e-6,
    carry_q: bool = False,
) -> AdversarialArbReport:
    """Re-check a fitted model on a fine, forward-refined float64 grid.

    Args:
        model: prices via ``forward(K, T, S, r, q, context) -> {'price'}``.
        S, r, q: snapshot spot / rate / dividend yield.
        T_grid: maturities to test (default a 1w..2y spread).
        context: per-snapshot context row, shape (context_dim,) or None.
        n_base / n_refine: base and per-forward refinement point counts.
        span: half-width of the base strike grid as a fraction of S.
        refine_span: half-width of the forward cluster as a fraction of F_T.
        tol: violation tolerance on second/first differences (float64).
        carry_q: if True, check calendar on e^{(r-q)T} C (the surrogate the real
            study currently uses); if False, on e^{rT} C (the quantity the
            architecture actually guarantees with the bump off). Default False.

    Returns:
        AdversarialArbReport.
    """
    if T_grid is None:
        T_grid = torch.tensor([0.02, 0.08, 0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
    T_grid = T_grid.to(torch.float64)

    was_training = model.training
    orig_dtype = next(model.parameters()).dtype
    model.eval().double()
    try:
        K = _refined_strike_grid(S, r, q, T_grid, n_base, n_refine, span, refine_span)
        n_K = K.shape[0]
        n_T = T_grid.shape[0]
        ctx = None
        if context is not None:
            ctx = context.to(torch.float64).view(1, -1)

        # Evaluate the (n_T, n_K) price surface row by row.
        C = torch.empty(n_T, n_K, dtype=torch.float64)
        delta = torch.empty(n_T, n_K, dtype=torch.float64)
        for i, T in enumerate(T_grid):
            Ti = torch.full((n_K,), float(T), dtype=torch.float64)
            Si = torch.full((n_K,), float(S), dtype=torch.float64)
            ri = torch.full((n_K,), float(r), dtype=torch.float64)
            qi = torch.full((n_K,), float(q), dtype=torch.float64)
            ctx_i = ctx.expand(n_K, -1) if ctx is not None else None
            price = model(K, Ti, Si, ri, qi, ctx_i)["price"]
            C[i] = price
            F_T = S * float(torch.exp(torch.tensor((r - q) * float(T))))
            intrinsic_fwd = torch.clamp(F_T - K, min=0.0)
            delta[i] = price * torch.exp(torch.tensor(r * float(T))) - intrinsic_fwd

        # --- Butterfly: divided second difference of C wrt K, per T row.
        K_l, K_m, K_r = K[:-2], K[1:-1], K[2:]
        h1 = (K_m - K_l).clamp(min=1e-12)
        h2 = (K_r - K_m).clamp(min=1e-12)
        slope_lo = (C[:, 1:-1] - C[:, :-2]) / h1
        slope_hi = (C[:, 2:] - C[:, 1:-1]) / h2
        second = (slope_hi - slope_lo) / ((h1 + h2) / 2.0)
        bf_viol = torch.relu(-second)               # (n_T, n_K-2)
        bf_mask = bf_viol > tol
        bf_count = int(bf_mask.sum())
        bf_worst = float(bf_viol.max()) if bf_viol.numel() else 0.0
        if bf_count > 0:
            idx = torch.argmax(bf_viol)
            ti, ki = divmod(int(idx), bf_viol.shape[1])
            worst_K = float(K_m[ki])
            worst_T = float(T_grid[ti])
        else:
            worst_K = worst_T = 0.0

        # --- Calendar (PRIMARY): at fixed FORWARD-MONEYNESS kappa = K / F_T.
        # The economically standard no-calendar-arb condition is total implied
        # variance non-decreasing in T at fixed forward log-moneyness; equivalently
        # the normalized undiscounted call nc(kappa,T) = e^{rT} C(kappa F_T, T)/F_T
        # is non-decreasing in T. (Fixed-STRIKE e^{rT}C is a different surrogate and
        # is reported separately below -- using it on a forward-anchored model such
        # as DensityNet manufactures spurious violations; this is the H5 issue.)
        kappa = _moneyness_grid(span, refine_span, n_base, n_refine)
        nKa = kappa.shape[0]
        nc = torch.empty(n_T, nKa, dtype=torch.float64)
        for i, T in enumerate(T_grid):
            F_T = S * float(torch.exp(torch.tensor((r - q) * float(T))))
            Ki = kappa * F_T
            Ti = torch.full((nKa,), float(T), dtype=torch.float64)
            Si = torch.full((nKa,), float(S), dtype=torch.float64)
            ri = torch.full((nKa,), float(r), dtype=torch.float64)
            qi = torch.full((nKa,), float(q), dtype=torch.float64)
            ctx_i = ctx.expand(nKa, -1) if ctx is not None else None
            pr = model(Ki, Ti, Si, ri, qi, ctx_i)["price"]
            nc[i] = pr * float(torch.exp(torch.tensor(r * float(T)))) / F_T   # normalized, O(1)
        cal_viol = torch.relu(nc[:-1, :] - nc[1:, :])           # non-decreasing in T at fixed kappa
        cal_count = int((cal_viol > tol).sum())                 # nc is O(1) -> tol is relative
        cal_worst = float(cal_viol.max()) if cal_viol.numel() else 0.0

        # --- Calendar (REFERENCE): fixed-strike e^{rT} C non-decreasing in T.
        carry_rate = (r - q) if carry_q else r
        Cw = C * torch.exp(carry_rate * T_grid).view(n_T, 1)
        cal_s_viol = torch.relu(Cw[:-1, :] - Cw[1:, :]) / float(S)   # normalize by spot
        cal_s_count = int((cal_s_viol > tol).sum())
        cal_s_worst = float(cal_s_viol.max()) if cal_s_viol.numel() else 0.0

        # --- A4: Delta >= 0, measured RELATIVE to spot (price-level: an absolute
        # 1e-6 INR tol at Nifty scale ~1e4 flags float noise, not arbitrage).
        a4_viol = torch.relu(-delta) / float(S)
        a4_count = int((a4_viol > tol).sum())
        a4_worst = float(a4_viol.max()) if a4_viol.numel() else 0.0
    finally:
        model.to(orig_dtype)
        if was_training:
            model.train()

    return AdversarialArbReport(
        n_T=n_T, n_K=n_K,
        butterfly_count=bf_count, calendar_count=cal_count, a4_count=a4_count,
        butterfly_worst=bf_worst, calendar_worst=cal_worst, a4_worst=a4_worst,
        worst_butterfly_K=worst_K, worst_butterfly_T=worst_T, tol=tol,
        calendar_strike_count=cal_s_count, calendar_strike_worst=cal_s_worst,
    )
