"""Full static no-arbitrage compliance diagnostic (beyond A1-A4).

The study's existing reports test butterfly (A1), calendar (A2), the expiry
boundary (A3), the forward-intrinsic lower bound (A4) and the monitored
Roger Lee tail slope. They do NOT test the remaining conditions in the
classical characterisation (Roper 2010; Cousot 2007; Carr-Madan 2005):

  (M)  K -> C(K, T) non-increasing:              dC/dK <= 0
  (D)  call-spread / digital bound:              -dC/dK <= e^{-rT}
       (equivalently 0 <= e^{rT} (-dC/dK) <= 1: the implied digital is a
        probability)
  (U)  upper bound / K -> 0 boundary:            C(K, T) <= S e^{-qT},
       with equality only at K = 0 (C(0, T) = S e^{-qT}, the discounted
       forward)

For ArbNet these can fail *architecturally*: its forward time value
Delta_theta(., T) is convex, strictly positive, and non-vanishing as
K -> infinity, so e^{rT} C = (F_T - K)^+ + Delta is eventually INCREASING in
K (a call-spread arbitrage) and C(0, T) = e^{-rT}(F_T + Delta(0,T)) EXCEEDS
the discounted forward (a covered-call arbitrage: sell the K=0 call, buy
e^{-qT} shares). A convex non-negative function on [0, inf) vanishing at 0
and at infinity is identically zero, so no nonzero convex Delta can satisfy
all the boundary conditions -- the violation is structural, not a fitting
accident. This module measures WHERE it bites on a fitted model.

Everything is float64. Violations are reported per moneyness bucket:

  in_band:  |K/S - 1| <= 0.30   (the traded band used by the paper's grid)
  wing:     0.30 < |K/S - 1| <= 0.60
  far:      K/S in [k_min, 4.0] outside the above (extrapolation region)

plus per-maturity structural summaries: the smallest K/S at which the call
becomes increasing (`mono_first_bite_K_over_S`), the K -> 0 boundary excess
(C(k_min S, T) - S e^{-qT})/S, the implied left-edge total mass
e^{rT}(-dC/dK) at the smallest strike (must be <= 1 for a probability
measure), and the right-edge forward slope e^{rT} dC/dK at the largest
strike (must be <= 0; positive = mass beyond every strike, ArbNet's
`rho_T` total mass exceeding 1).

Works for any model exposing forward(K, T, S, r, q, context) -> {'price'}.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import math

import torch


_BUCKETS = ("in_band", "wing", "far")


@dataclass
class MaturityCompliance:
    T: float
    # (M) monotonicity: count of grid intervals with dC/dK > tol, per bucket
    mono_counts: Dict[str, int]
    mono_worst_slope: float            # max dC/dK (dimensionless, <= 0 if clean)
    mono_first_bite_K_over_S: float    # smallest K/S midpoint with dC/dK > tol (nan if none)
    # (D) digital bound: count of intervals with e^{rT}(-dC/dK) > 1 + tol
    digital_counts: Dict[str, int]
    digital_worst: float               # max e^{rT}(-dC/dK) - 1
    # (U) upper bound: count of grid points with C > S e^{-qT} (rel tol on S)
    upper_counts: Dict[str, int]
    upper_worst_rel: float             # max (C - S e^{-qT})/S
    # K -> 0 boundary: (C(k_min S, T) - S e^{-qT}) / S.  > 0 is a covered-call
    # arbitrage of that relative size at the smallest tested strike.
    boundary_gap_rel: float
    # edge slopes of the forward call e^{rT} C:
    left_edge_mass: float              # e^{rT}(-dC/dK) at K = k_min S (<= 1 if a prob.)
    right_edge_fwd_slope: float        # e^{rT} dC/dK at K = k_max S (<= 0 required)


@dataclass
class FullComplianceReport:
    n_T: int
    n_K: int
    k_min: float                       # smallest tested K/S
    k_max: float                       # largest tested K/S
    tol: float
    # aggregate counts over all maturities, per bucket
    mono_counts: Dict[str, int]
    digital_counts: Dict[str, int]
    upper_counts: Dict[str, int]
    mono_worst_slope: float
    digital_worst: float
    upper_worst_rel: float
    boundary_gap_rel_worst: float
    right_edge_fwd_slope_worst: float
    left_edge_mass_worst: float
    per_maturity: List[MaturityCompliance] = field(default_factory=list)

    @property
    def clean_in_band(self) -> bool:
        return (self.mono_counts["in_band"] == 0
                and self.digital_counts["in_band"] == 0
                and self.upper_counts["in_band"] == 0)

    @property
    def clean_everywhere(self) -> bool:
        return (sum(self.mono_counts.values()) == 0
                and sum(self.digital_counts.values()) == 0
                and sum(self.upper_counts.values()) == 0
                and self.boundary_gap_rel_worst <= self.tol
                and self.right_edge_fwd_slope_worst <= self.tol)

    def as_record(self, prefix: str = "full") -> Dict:
        """Flatten for a per-day study record."""
        return {
            f"{prefix}_mono_inband": self.mono_counts["in_band"],
            f"{prefix}_mono_wing": self.mono_counts["wing"],
            f"{prefix}_mono_far": self.mono_counts["far"],
            f"{prefix}_digital_inband": self.digital_counts["in_band"],
            f"{prefix}_digital_wing": self.digital_counts["wing"],
            f"{prefix}_digital_far": self.digital_counts["far"],
            f"{prefix}_upper_inband": self.upper_counts["in_band"],
            f"{prefix}_upper_wing": self.upper_counts["wing"],
            f"{prefix}_upper_far": self.upper_counts["far"],
            f"{prefix}_mono_worst_slope": self.mono_worst_slope,
            f"{prefix}_digital_worst": self.digital_worst,
            f"{prefix}_upper_worst_rel": self.upper_worst_rel,
            f"{prefix}_boundary_gap_rel_worst": self.boundary_gap_rel_worst,
            f"{prefix}_right_edge_fwd_slope_worst": self.right_edge_fwd_slope_worst,
            f"{prefix}_left_edge_mass_worst": self.left_edge_mass_worst,
            f"{prefix}_first_bite_K_over_S_min": min(
                (m.mono_first_bite_K_over_S for m in self.per_maturity
                 if math.isfinite(m.mono_first_bite_K_over_S)),
                default=float("nan"),
            ),
        }

    def __str__(self) -> str:
        fb = [m.mono_first_bite_K_over_S for m in self.per_maturity
              if math.isfinite(m.mono_first_bite_K_over_S)]
        fb_s = f"{min(fb):.3f}" if fb else "none"
        return (
            f"Full compliance on {self.n_T}x{self.n_K} float64 grid, "
            f"K/S in [{self.k_min:.4f}, {self.k_max:.1f}] (tol={self.tol:.1e}):\n"
            f"  mono (dC/dK<=0):      in_band {self.mono_counts['in_band']}, "
            f"wing {self.mono_counts['wing']}, far {self.mono_counts['far']} "
            f"(worst slope {self.mono_worst_slope:+.3e}; first bite K/S={fb_s})\n"
            f"  digital (<=e^-rT):    in_band {self.digital_counts['in_band']}, "
            f"wing {self.digital_counts['wing']}, far {self.digital_counts['far']} "
            f"(worst excess {self.digital_worst:+.3e})\n"
            f"  upper (C<=Se^-qT):    in_band {self.upper_counts['in_band']}, "
            f"wing {self.upper_counts['wing']}, far {self.upper_counts['far']} "
            f"(worst rel {self.upper_worst_rel:+.3e})\n"
            f"  K->0 boundary gap:    worst rel {self.boundary_gap_rel_worst:+.3e}\n"
            f"  edge masses:          left e^rT(-C') worst {self.left_edge_mass_worst:.4f} "
            f"(<=1), right e^rT C' worst {self.right_edge_fwd_slope_worst:+.3e} (<=0)\n"
            f"  -> in-band {'CLEAN' if self.clean_in_band else 'VIOLATIONS'}; "
            f"everywhere {'CLEAN' if self.clean_everywhere else 'VIOLATIONS'}"
        )


def _grid_K_over_S(k_min: float, k_max: float, band: float, wing: float,
                   n_low: int, n_band: int, n_far: int) -> torch.Tensor:
    """K/S grid: geometric below the band, dense linear across +-wing, geometric out to k_max."""
    lo = torch.exp(torch.linspace(math.log(k_min), math.log(1.0 - wing), n_low,
                                  dtype=torch.float64))
    mid = torch.linspace(1.0 - wing, 1.0 + wing, n_band, dtype=torch.float64)
    hi = torch.exp(torch.linspace(math.log(1.0 + wing), math.log(k_max), n_far,
                                  dtype=torch.float64))
    g = torch.unique(torch.cat([lo, mid, hi]), sorted=True)
    # min spacing guard (same rationale as adversarial._dedupe_sorted)
    keep = torch.ones_like(g, dtype=torch.bool)
    last = float(g[0])
    for i in range(1, g.numel()):
        if float(g[i]) - last > 1e-7:
            last = float(g[i])
        else:
            keep[i] = False
    return g[keep]


def _bucket_of(k_over_s: torch.Tensor, band: float, wing: float) -> torch.Tensor:
    """0 = in_band, 1 = wing, 2 = far."""
    d = (k_over_s - 1.0).abs()
    out = torch.full_like(k_over_s, 2.0)
    out[d <= wing] = 1.0
    out[d <= band] = 0.0
    return out.long()


@torch.no_grad()
def full_compliance_report(
    model: torch.nn.Module,
    S: float,
    r: float,
    q: float = 0.0,
    T_grid: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    k_min: float = 1e-3,
    k_max: float = 4.0,
    band: float = 0.30,
    wing: float = 0.60,
    n_low: int = 120,
    n_band: int = 601,
    n_far: int = 240,
    tol: float = 1e-6,
) -> FullComplianceReport:
    """Test monotonicity, the digital bound, the upper bound and the K->0
    boundary on a fitted model, bucketed by moneyness.

    Args:
        model: prices via forward(K, T, S, r, q, context) -> {'price'}.
        S, r, q: snapshot spot / risk-free rate / dividend yield.
        T_grid: maturities to test (default: the adversarial module's spread).
        context: per-snapshot context row, shape (context_dim,) or None.
        k_min, k_max: tested K/S range (k_min ~ 0 probes the boundary).
        band, wing: bucket edges on |K/S - 1|.
        tol: violation tolerance (slopes are dimensionless O(1); tol is
            effectively relative).
    """
    if T_grid is None:
        T_grid = torch.tensor([0.02, 0.08, 0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
    T_grid = T_grid.to(torch.float64)

    was_training = model.training
    orig_dtype = next(model.parameters()).dtype
    model.eval().double()
    try:
        ks = _grid_K_over_S(k_min, k_max, band, wing, n_low, n_band, n_far)
        K = ks * float(S)
        n_K = K.shape[0]
        n_T = T_grid.shape[0]
        ctx = None
        if context is not None:
            ctx = context.to(torch.float64).view(1, -1)

        # interval midpoints -> buckets for the slope-based checks
        ks_mid = 0.5 * (ks[:-1] + ks[1:])
        b_mid = _bucket_of(ks_mid, band, wing)
        b_pt = _bucket_of(ks, band, wing)

        mono_counts = {b: 0 for b in _BUCKETS}
        digital_counts = {b: 0 for b in _BUCKETS}
        upper_counts = {b: 0 for b in _BUCKETS}
        mono_worst = -math.inf
        digital_worst = -math.inf
        upper_worst = -math.inf
        boundary_worst = -math.inf
        right_edge_worst = -math.inf
        left_mass_worst = -math.inf
        per_mat: List[MaturityCompliance] = []

        dK = (K[1:] - K[:-1]).clamp(min=1e-12)
        for i, T in enumerate(T_grid):
            Tf = float(T)
            Ti = torch.full((n_K,), Tf, dtype=torch.float64)
            Si = torch.full((n_K,), float(S), dtype=torch.float64)
            ri = torch.full((n_K,), float(r), dtype=torch.float64)
            qi = torch.full((n_K,), float(q), dtype=torch.float64)
            ctx_i = ctx.expand(n_K, -1) if ctx is not None else None
            C = model(K, Ti, Si, ri, qi, ctx_i)["price"]

            disc = math.exp(-r * Tf)
            growth = math.exp(r * Tf)
            upper_bound = float(S) * math.exp(-q * Tf)

            slope = (C[1:] - C[:-1]) / dK                    # dC/dK on intervals

            # (M) monotonicity
            mono_viol = slope > tol
            m_counts = {b: int((mono_viol & (b_mid == j)).sum())
                        for j, b in enumerate(_BUCKETS)}
            m_worst = float(slope.max())
            bite_idx = mono_viol.nonzero()
            first_bite = float(ks_mid[bite_idx[0, 0]]) if bite_idx.numel() else float("nan")

            # (D) digital bound: e^{rT}(-slope) <= 1
            digital = growth * (-slope)
            d_excess = digital - 1.0
            d_viol = d_excess > tol
            d_counts = {b: int((d_viol & (b_mid == j)).sum())
                        for j, b in enumerate(_BUCKETS)}
            d_worst = float(d_excess.max())

            # (U) upper bound (pointwise), relative to S
            u_rel = (C - upper_bound) / float(S)
            u_viol = u_rel > tol
            u_counts = {b: int((u_viol & (b_pt == j)).sum())
                        for j, b in enumerate(_BUCKETS)}
            u_worst = float(u_rel.max())

            gap0 = float((C[0] - upper_bound) / float(S))
            left_mass = float(growth * (-(C[1] - C[0]) / dK[0]))
            right_fwd_slope = float(growth * (C[-1] - C[-2]) / dK[-1])

            per_mat.append(MaturityCompliance(
                T=Tf, mono_counts=m_counts, mono_worst_slope=m_worst,
                mono_first_bite_K_over_S=first_bite,
                digital_counts=d_counts, digital_worst=d_worst,
                upper_counts=u_counts, upper_worst_rel=u_worst,
                boundary_gap_rel=gap0, left_edge_mass=left_mass,
                right_edge_fwd_slope=right_fwd_slope,
            ))
            for b in _BUCKETS:
                mono_counts[b] += m_counts[b]
                digital_counts[b] += d_counts[b]
                upper_counts[b] += u_counts[b]
            mono_worst = max(mono_worst, m_worst)
            digital_worst = max(digital_worst, d_worst)
            upper_worst = max(upper_worst, u_worst)
            boundary_worst = max(boundary_worst, gap0)
            right_edge_worst = max(right_edge_worst, right_fwd_slope)
            left_mass_worst = max(left_mass_worst, left_mass)
    finally:
        model.to(orig_dtype)
        if was_training:
            model.train()

    return FullComplianceReport(
        n_T=n_T, n_K=n_K, k_min=float(ks[0]), k_max=float(ks[-1]), tol=tol,
        mono_counts=mono_counts, digital_counts=digital_counts,
        upper_counts=upper_counts,
        mono_worst_slope=mono_worst, digital_worst=digital_worst,
        upper_worst_rel=upper_worst, boundary_gap_rel_worst=boundary_worst,
        right_edge_fwd_slope_worst=right_edge_worst,
        left_edge_mass_worst=left_mass_worst,
        per_maturity=per_mat,
    )
