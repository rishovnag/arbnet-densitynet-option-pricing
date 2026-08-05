"""Throwaway numpy prototype for the localized concave-bump reformulation.

Purpose (per ArbNet_critique_and_path_forward.docx, section 4.4): before touching
the torch architecture, check on a single Nifty-scale slice whether adding a
mass-budgeted concave-centered bump to the single-cone correction

    (a) flattens the Dirac density atom at the forward, and
    (b) how much pointwise butterfly headroom the (dagger) mass budget leaves.

Pure numpy, so it runs without torch. It does NOT use the real model; it builds a
synthetic but realistic single-cone time value and the bump exactly as the torch
implementation will, so the math (and the sign) is validated here first.

Sign note: Breeden-Litzenberger gives the risk-neutral density
    q(K) = delta(K - F) + d2/dK2 Delta(K).
The intrinsic kink deposits a unit atom +delta(K-F). To cancel/flatten it we need
NEGATIVE curvature in Delta near F, i.e. we ADD a non-negative concave-centered
bump (raised cosine). The doc's literal "Delta = Dcvx - m*b with b concave" has
the wrong sign (it deepens the atom) and makes A4 (Delta >= 0) a worry; ADDING a
non-negative bump fixes both.
"""
from __future__ import annotations

import numpy as np

# Raised cosine on [-1, 1]:
#   phi(x)   = (1 + cos(pi x)) / 2          phi(0)=1, phi(+-1)=0, phi >= 0
#   phi''(x) = -(pi^2/2) cos(pi x)          < 0 for |x|<1/2 (concave core)
#                                           > 0 for 1/2<|x|<1 (convex tails)
# Negative-curvature mass of phi in x units:
#   int_{-1/2}^{1/2} (-phi'') dx = pi.
PHI_NEG_CURV_MASS = np.pi


def phi(x):
    out = np.zeros_like(x)
    m = np.abs(x) <= 1.0
    out[m] = 0.5 * (1.0 + np.cos(np.pi * x[m]))
    return out


def single_cone_time_value(K, S, scale, curv):
    """Convex, non-negative, V-shaped time value mimicking the ICNN mixture."""
    x = (K / S - 1.0)
    sp = lambda z: np.logaddexp(0.0, z)
    return scale * S * (sp(curv * x) + sp(-curv * x))


def bump_amplitude(sigma, gate):
    """K-space bump amplitude M with the (dagger) mass budget.

    Negative-curvature mass of M*phi((K-F)/sigma) in K is (M/sigma)*pi.
    Require <= 1 (the intrinsic unit atom budget): M = (sigma/pi)*gate, gate<1.
    """
    return (sigma / PHI_NEG_CURV_MASS) * gate


def _measure(K, F, h, intrinsic, Dcvx, disc, sigma, gate, fwd_cell):
    M = bump_amplitude(sigma, gate)
    bump = M * phi((K - F) / sigma)
    Delta = Dcvx + bump                       # ADD (correct sign)
    C = disc * (intrinsic + Delta)
    d2C = np.zeros_like(C)
    d2C[1:-1] = (C[2:] - 2.0 * C[1:-1] + C[:-2]) / (h * h)
    atom = (d2C[fwd_cell] / disc)             # density at the forward cell
    mask = np.ones_like(d2C, dtype=bool)
    mask[fwd_cell - 1: fwd_cell + 2] = False  # exclude the atom cell + neighbours
    viol = max(0.0, -(d2C[mask].min()))
    a4_ok = bool(np.all(Delta >= -1e-9))
    return M, atom, viol, a4_ok


def run(S=20000.0, r=0.065, q=0.012, T=0.10, scale=0.004, curv=2.0,
        gate=0.95, sigma_cells=(2000, 600, 200, 60, 20), n=8001, span=0.5):
    K = np.linspace(S * (1 - span), S * (1 + span), n)
    h = K[1] - K[0]
    F_target = S * np.exp((r - q) * T)
    fwd_cell = int(np.argmin(np.abs(K - F_target)))
    F = K[fwd_cell]                            # snap F onto a node: clean 1-cell atom

    Dcvx = single_cone_time_value(K, S, scale, curv)
    intrinsic = np.maximum(F - K, 0.0)
    disc = np.exp(-r * T)

    _, atom0, _, _ = _measure(K, F, h, intrinsic, Dcvx, disc, h, 0.0, fwd_cell)
    print("S=%.0f  F=%.1f (on grid)  T=%s  grid h=%.2f" % (S, F, T, h))
    print("ATM single-cone time value Dcvx(F) ~ %.1f INR" % np.interp(F, K, Dcvx))
    print("Bare forward-cell density (the atom), no bump: %.4f" % atom0)
    print("Fixed mass gate = %s (negative-curvature mass = %s < 1 unit)\n" % (gate, gate))

    print("%9s %7s %9s %10s %10s %16s %5s" %
          ("sigma", "cells", "M(INR)", "atom dens", "atom drop", "worst bfly viol", "A4"))
    print("-" * 72)
    for sc in sigma_cells:
        sigma = sc * h
        M, atom, viol, a4 = _measure(K, F, h, intrinsic, Dcvx, disc, sigma, gate, fwd_cell)
        drop = 1.0 - atom / atom0
        print("%9.0f %7d %9.1f %10.4f %9.1f%% %16.2e %5s" %
              (sigma, sc, M, atom, 100 * drop, viol, a4))

    print()
    print("Reading (the gap-(1) tradeoff, at a FIXED in-budget mass gate < 1):")
    print(" - WIDE bump (top): negligible butterfly violation, but barely dents")
    print("   the atom; a wide cap cannot cancel a 1-cell spike.")
    print(" - NARROW bump (bottom): flattens the atom, but the pointwise negative")
    print("   curvature ~ pi/(2 sigma) overshoots Dcvx local convexity and makes a")
    print("   large butterfly violation right beside the forward.")
    print(" => the (dagger) mass budget is necessary but NOT sufficient for")
    print("    pointwise butterfly. Compliance is no longer free: it must be")
    print("    checked on a grid FINER than training and refined near K=F, else a")
    print("    coarse grid reports a false zero. A4 (Delta>=0) is free (added).")


if __name__ == "__main__":
    run()
