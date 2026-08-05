"""Prototype for way-forward direction #4: a DIRECT convex / risk-neutral-density
parametrization of the call surface, vs ArbNet's intrinsic-plus-convex form.

Hypothesis (single maturity): the Dirac density atom at the forward is an
artifact of writing  C = e^{-rT}[(F-K)^+ + Delta]  with Delta convex -- the relu
intrinsic deposits delta(K-F) in the density. If instead we parametrize the call
as the discounted payoff under a learned NON-NEGATIVE density that integrates to 1
and has mean = forward (a martingale), then by construction:
  * butterfly holds exactly      (C'' = e^{-rT} q >= 0, q the density),
  * A4 holds automatically       (C = e^{-rT}E[(X-K)^+] >= e^{-rT}(F-K)^+, Jensen),
  * the density is smooth/bell    (no atom),
and the expressivity wall that costs ArbNet ~50-60 INR RMSE should largely go away.

This script tests that on a realistic single snapshot (SVI smile, Nifty scale),
in pure numpy/scipy. We represent the density as a non-negative mixture of
lognormals (each component is itself a martingale forward), fit by NNLS with
soft normalization + martingale constraints, and compare against the best
intrinsic-plus-convex (ArbNet-style) fit.

It also does a quick 3-maturity calendar diagnostic, which is the part that is
NOT free for the direct form (the open cross-maturity 'convex order' condition).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss  # noqa: F401 (kept for reference)
from scipy.optimize import nnls
from scipy.stats import norm


# ---------------------------------------------------------------------------
def black_call_undisc(F, K, totvar):
    """Undiscounted Black call with forward F and total variance totvar (=sig^2 T)."""
    F = np.asarray(F, float); K = np.asarray(K, float)
    v = np.sqrt(np.maximum(totvar, 1e-16))
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def lognormal_pdf(x, F, totvar):
    """pdf of X with E[X]=F and total variance totvar (lognormal)."""
    v = np.sqrt(max(totvar, 1e-16))
    mu = np.log(F) - 0.5 * v * v
    return np.exp(-(np.log(x) - mu) ** 2 / (2 * v * v)) / (x * v * np.sqrt(2 * np.pi))


def svi_total_var(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def second_diff(C, K):
    h1 = K[1:-1] - K[:-2]; h2 = K[2:] - K[1:-1]
    slo = (C[1:-1] - C[:-2]) / h1; shi = (C[2:] - C[1:-1]) / h2
    return (shi - slo) / ((h1 + h2) / 2.0)


# ---------------------------------------------------------------------------
def fit_density_mixture(K, C_und_mkt, F, n_f=25, n_v=12, lam_con=1e3):
    """Method B: non-negative lognormal mixture with soft sum=1 and mean=F."""
    f_grid = F * np.exp(np.linspace(-0.45, 0.45, n_f))
    v_grid = np.linspace(0.03, 0.9, n_v)            # total vol = sigma*sqrt(T)
    comps = [(f, v) for f in f_grid for v in v_grid]
    A = np.column_stack([black_call_undisc(f, K, v * v) for (f, v) in comps])  # (nK, nComp)
    # constraint rows: sum(w)=1 (mass) and sum(w*f)=F (martingale), weighted by lam_con
    ones = np.ones((1, len(comps)))
    fs = np.array([[f for (f, v) in comps]])
    A_aug = np.vstack([A, lam_con * ones, lam_con * fs])
    b_aug = np.concatenate([C_und_mkt, [lam_con * 1.0], [lam_con * F]])
    w, _ = nnls(A_aug, b_aug)
    C_und = A @ w
    return w, comps, C_und


def fit_intrinsic_plus_convex(K, C_und_mkt, F, n_basis=40, sharp=4.0):
    """Method A (ArbNet-style): C_und = (F-K)^+ + Delta, Delta convex & >=0.

    Delta = sum_m beta_m * softplus(sharp*(K-kappa_m)/F)*(F/sharp) + ...,
    using BOTH increasing and decreasing smooth-convex hinges so Delta can grow on
    either wing; beta>=0 keeps Delta convex and non-negative. Density contribution
    Delta'' is smooth and >=0, so the model's density is delta(K-F) [the atom] plus
    a smooth non-negative part -- exactly ArbNet's structural form.
    """
    intrinsic = np.maximum(F - K, 0.0)
    tv = C_und_mkt - intrinsic                     # target time value (undiscounted)
    kappa = np.linspace(K.min(), K.max(), n_basis)
    cols = []
    for kp in kappa:
        cols.append(np.logaddexp(0.0, sharp * (K - kp) / F) * (F / sharp))   # up hinge
        cols.append(np.logaddexp(0.0, -sharp * (K - kp) / F) * (F / sharp))  # down hinge
    cols.append(np.ones_like(K))                    # non-negative level
    A = np.column_stack(cols)
    beta, _ = nnls(A, tv)
    Delta = A @ beta
    C_und = intrinsic + Delta
    return C_und, Delta


# ---------------------------------------------------------------------------
def run(S=20000.0, r=0.065, q=0.012, T=0.25,
        svi=(0.005, 0.045, -0.6, 0.0, 0.10), n_K=61):
    F = S * np.exp((r - q) * T)
    disc = np.exp(-r * T)
    K = F * np.exp(np.linspace(-0.35, 0.35, n_K))
    k = np.log(K / F)

    # --- "market": arbitrage-free SVI smile -> Black prices --------------------
    w_mkt = svi_total_var(k, *svi)
    C_und_mkt = black_call_undisc(F, K, w_mkt)      # undiscounted
    C_mkt = disc * C_und_mkt
    q_mkt = second_diff(C_und_mkt, K)               # undiscounted density (interior)
    assert q_mkt.min() > -1e-10, "market itself has butterfly arb; retune SVI"

    # --- Method B: direct martingale density mixture ---------------------------
    wB, comps, C_undB = fit_density_mixture(K, C_und_mkt, F)
    C_B = disc * C_undB
    massB = wB.sum(); meanB = float(np.sum(wB * np.array([f for f, v in comps])))
    qB = np.array([np.sum([wB[i] * lognormal_pdf(x, comps[i][0], comps[i][1] ** 2)
                           for i in range(len(comps))]) for x in K])
    rmseB = float(np.sqrt(np.mean((C_B - C_mkt) ** 2)))
    bflyB = float(second_diff(C_undB, K).min())
    a4B = float((C_undB - np.maximum(F - K, 0.0)).min())

    # --- Method A: intrinsic + convex Delta (ArbNet structural form) -----------
    C_undA, DeltaA = fit_intrinsic_plus_convex(K, C_und_mkt, F)
    C_A = disc * C_undA
    rmseA = float(np.sqrt(np.mean((C_A - C_mkt) ** 2)))
    bflyA = float(second_diff(C_undA, K).min())
    a4A = float(DeltaA.min())                        # Delta>=0 is A4 here
    # ATM-region residual (where the V-smile / atom hurts most): |k|<5%
    atm = np.abs(k) < 0.05
    atm_resA = float(np.sqrt(np.mean((C_A[atm] - C_mkt[atm]) ** 2)))
    atm_resB = float(np.sqrt(np.mean((C_B[atm] - C_mkt[atm]) ** 2)))

    fwd_cell = int(np.argmin(np.abs(K - F)))
    print(f"S={S:.0f}  F={F:.1f}  T={T}  ATM IV={np.sqrt(w_mkt[fwd_cell]/T):.1%}  "
          f"n_strikes={n_K}\n")
    print(f"{'':28s}{'Method A (intrinsic+cvx)':>26s}{'Method B (density)':>22s}")
    print("-" * 76)
    print(f"{'price RMSE (INR)':28s}{rmseA:>26.2f}{rmseB:>22.2f}")
    print(f"{'ATM-region price RMSE':28s}{atm_resA:>26.2f}{atm_resB:>22.2f}")
    print(f"{'butterfly min 2nd-diff':28s}{bflyA:>26.2e}{bflyB:>22.2e}")
    print(f"{'A4 slack (>=0 ok)':28s}{a4A:>26.3f}{a4B:>22.3f}")
    print(f"{'density mass / mean-F':28s}{'delta_F atom + smooth':>26s}"
          f"{f'{massB:.3f} / {meanB-F:+.1f}':>22s}")
    # atom indicator: discrete density at the forward cell (undiscounted 2nd diff)
    d2A = second_diff(C_undA, K); d2B = second_diff(C_undB, K)
    iF = fwd_cell - 1
    print(f"{'density@F vs neighbour avg':28s}"
          f"{d2A[iF]/ (0.5*(d2A[iF-2]+d2A[iF+2])+1e-12):>26.1f}"
          f"{d2B[iF]/ (0.5*(d2B[iF-2]+d2B[iF+2])+1e-12):>22.1f}")
    print("\n(density@F ratio >> 1 means a spike/atom at the forward; ~1 means smooth)")

    # --- calendar diagnostic for Method B across 3 maturities ------------------
    print("\nCalendar check for Method B (independent fits per maturity):")
    Ts = [0.08, 0.25, 0.75]
    C_cols = []
    Kc = F * np.exp(np.linspace(-0.3, 0.3, 41))     # common strike grid
    for Ti in Ts:
        Fi = S * np.exp((r - q) * Ti)
        wi = svi_total_var(np.log(Kc / Fi), *svi) * (Ti / T)   # scale var with T
        Cundi = black_call_undisc(Fi, Kc, wi)
        wB_i, comps_i, CundB_i = fit_density_mixture(Kc, Cundi, Fi)
        C_cols.append(CundB_i)                       # e^{rT}C = undiscounted call
    Cmat = np.array(C_cols)                          # (3, nK), rows = maturities
    cal_viol = np.maximum(Cmat[:-1] - Cmat[1:], 0.0)  # e^{rT}C should be nondecr in T
    print(f"  e^rT*C non-decreasing in T?  max calendar violation = "
          f"{cal_viol.max():.3e}  (0 = clean)")
    print("  (this is the cross-maturity 'convex order' condition -- the open part)")


if __name__ == "__main__":
    run()
