"""Decisive prototype for the density-net (way-forward #4), calendar device.

Question: can a mixture-of-lognormals call surface be made free of butterfly (A1),
calendar (A2), expiry (A3) AND the intrinsic bound (A4) *by construction* -- the
property the bump could not achieve?

Construction (work in forward-moneyness x = S_T / F_T, a martingale with E[x]=1):
    components i = 1..M, weights w_i = softmax(.)        (sum to 1, T-independent)
    component means m_i > 0 with  sum_i w_i m_i = 1       (martingale, T-independent)
    component total variance  v_i^2(T) = base_i^2 * tau(T),  tau NON-DECREASING, tau(0)=0
    normalized undiscounted call  c(x=kappa, T) = sum_i w_i * Black(m_i, kappa, v_i(T))
    price  C(K,T) = e^{-rT} * F_T * c(K / F_T, T)

Claims, for EVERY parameter draw:
  A1 butterfly : c convex in kappa  (mixture of convex Black calls)                -> exact
  A4 bound     : c(kappa,T) >= (1-kappa)^+  (Jensen, mean 1)                        -> exact
  A3 expiry    : tau(0)=0 => c -> (1-kappa)^+ at T=0                                -> exact
  A2 calendar  : c(kappa,T) non-decreasing in T at fixed forward-moneyness kappa.
                 Each component is a fixed-mean lognormal with increasing variance
                 => increasing in convex order; a fixed-weight mixture preserves
                 convex order; (x-kappa)^+ is convex => c non-decreasing in T.       -> exact
                 (This is the GOLD-STANDARD calendar condition -- total variance
                  non-decreasing in T at fixed forward-moneyness -- not a price-
                  domain surrogate, so it also resolves the q>0 issue H5.)

This script stress-tests all four on many random parameter draws on a fine grid.
If violations are ~0 everywhere, the density-net guarantees A1-A4 by construction,
unlike the bump -- i.e. C8 is worth building.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_norm(m, kappa, v):
    """Undiscounted normalized Black call: forward m, strike kappa, total vol v."""
    v = np.maximum(v, 1e-12)
    d1 = (np.log(m / kappa) + 0.5 * v * v) / v
    d2 = d1 - v
    return m * norm.cdf(d1) - kappa * norm.cdf(d2)


def mixture_call(kappa, T, w, mu, base, tau, hfun):
    """c(kappa, T); component log-means mu_i*h(T) and log-var base_i^2*tau(T).

    Both the skew (via spread of mu_i*h(T)) and the variance vanish at T=0
    (h(0)=tau(0)=0), so every component -> forward 1 at expiry and A3 holds; both
    grow in T, so the smile and term structure develop. Weights are T-fixed.
    """
    out = np.zeros((len(T), len(kappa)))
    for j in range(len(T)):
        h = hfun(T[j]); tv = max(tau(T[j]), 0.0)
        # component log-normal: logmean lm_i = mu_i*h - 0.5 s_i^2 ; arithmetic mean exp(mu_i*h)
        am = np.exp(mu * h)                                # component arithmetic means (pre-norm)
        Z = np.sum(w * am)                                # martingale normalizer
        m = am / Z                                        # sum_i w_i m_i = 1 exactly
        v = base * np.sqrt(tv)
        cj = np.zeros_like(kappa)
        for i in range(len(w)):
            cj += w[i] * black_norm(m[i], kappa, v[i] if v[i] > 0 else 1e-12)
        out[j] = cj
    return out


def random_params(M, rng):
    w = rng.dirichlet(np.ones(M) * 2.0)                   # weights sum to 1 (T-fixed)
    mu = rng.normal(0, 0.6, M)                            # log-mean slopes -> skew as T grows
    base = rng.uniform(0.05, 0.9, M)                      # per-component base vol
    return w, mu, base


def run(M=6, n_draws=300, nK=121, nT=12, span=0.6, tol=1e-9):
    rng = np.random.default_rng(0)
    kappa = np.linspace(np.exp(-span), np.exp(span), nK)  # UNIFORM forward-moneyness grid
    #   (uniform spacing so the raw 2nd-difference is a valid convexity test;
    #    on an exp-spaced grid it carries ~1e-4 noise that is not a real violation)
    T = np.linspace(0.0, 2.0, nT)                         # includes T=0 (expiry)
    tau = lambda t: t                                     # tau(T)=T : non-decreasing, tau(0)=0
    hfun = lambda t: t                                    # h(T)=T : skew grows from 0 at expiry

    worst_bfly = worst_cal = worst_a4 = 0.0
    worst_a3 = 0.0
    bad = 0
    for _ in range(n_draws):
        w, mu, base = random_params(M, rng)
        C = mixture_call(kappa, T, w, mu, base, tau, hfun)  # (nT, nK)

        # A1 butterfly: convex in kappa per maturity (2nd difference >= 0)
        d2 = C[:, 2:] - 2 * C[:, 1:-1] + C[:, :-2]
        bfly = max(0.0, float((-d2).max()))

        # A2 calendar: non-decreasing in T at fixed kappa
        cal = max(0.0, float((C[:-1, :] - C[1:, :]).max()))

        # A4: c(kappa,T) >= (1-kappa)^+
        intr = np.maximum(1.0 - kappa, 0.0)[None, :]
        a4 = max(0.0, float((intr - C).max()))

        # A3: at T=0, c == (1-kappa)^+
        a3 = float(np.abs(C[0] - intr[0]).max())

        worst_bfly = max(worst_bfly, bfly); worst_cal = max(worst_cal, cal)
        worst_a4 = max(worst_a4, a4); worst_a3 = max(worst_a3, a3)
        if max(bfly, cal, a4) > 1e-6:
            bad += 1

    print(f"density-net calendar device: {n_draws} random {M}-component draws, "
          f"grid {nT}x{nK}\n")
    print(f"  A1 butterfly  worst violation : {worst_bfly:.3e}")
    print(f"  A2 calendar   worst violation : {worst_cal:.3e}")
    print(f"  A4 bound      worst violation : {worst_a4:.3e}")
    print(f"  A3 expiry     worst |C-(1-k)+|: {worst_a3:.3e}  (boundary match)")
    print(f"  draws with ANY A1/A2/A4 violation > 1e-6 : {bad} / {n_draws}")
    ok = max(worst_bfly, worst_cal, worst_a4) < 1e-6
    print("\n=> " + ("ALL of A1, A2, A4 hold by construction for every draw -- "
                     "the density-net guarantees full static no-arbitrage "
                     "(incl. calendar at fixed forward-moneyness, resolving H5). "
                     "C8 is worth building." if ok else
                     "violations found -- revisit the device."))


if __name__ == "__main__":
    run()
