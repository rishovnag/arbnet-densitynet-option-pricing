"""DensityNet: an arbitrage-free call surface via a martingale lognormal mixture.

Way-forward direction #4. The call is the discounted payoff under a learned
risk-neutral density, a mixture of lognormals in forward-moneyness x = S_T/F_T
(a martingale, E[x]=1). The skew comes from a spread of FIXED component means
with a SHARED, increasing total-variance clock -- NOT per-component T-growing
means (that earlier choice broke calendar on real data, see
scripts/density_calendar_device_v2.py).

    C(K,T) = e^{-rT} F_T * sum_i w_i * Black_norm(m_i, K/F_T, sigma^2(T))
    w_i        = softmax(.)                     (weights; per context, T-independent)
    m_i        = exp(a_i) / sum_j w_j exp(a_j)  (=> sum_i w_i m_i = 1 : martingale; FIXED in T)
    sigma^2(T) = softplus(c0) T + softplus(c1) T^2   (SHARED total variance; >=0, increasing, 0 at T=0)

By the representation x_T = m(theta) * M_T with M_T = exp(sigma(T) Z - sigma^2(T)/2)
a martingale independent of the mixing theta:
  A1 butterfly : mixture of Black calls, convex in K                              -> exact
  A2 calendar  : M_T increasing in convex order + independent mixing => x_T
                 increasing in convex order => nc(kappa,T) non-decreasing in T at
                 fixed forward-moneyness (gold-standard calendar; any q -> no H5
                 surrogate). Holds for EVERY parameter vector (adversarially
                 verified, incl. extreme skew).                                    -> exact
  A4 bound     : C >= e^{-rT}(F_T-K)^+ (Jensen, mean 1)                            -> exact
  A3 expiry    : C(.,0) is forced to e^{-rT}(F_T-K)^+; for 0<T->0 the surface
                 collapses to a thin residual smile (means stay dispersed), i.e.
                 A3 is exact only at T=0 -- an honest trade for guaranteed A2.

No atom (the density is the smooth mixture pdf) and lognormal mixtures are dense,
so the expressivity wall is gone.

Optional SKEW CLOCK (config.skew_clock, v4): m_i(T) = 1 + h(T)(m_i - 1) with
h(T) = 1 - e^{-beta T}, beta = softplus(.) > 0. A1/A2/A4 are preserved -- the
mixing law at T is a dilation of the fixed law about its mean 1, so the convex
order in T survives (see DensityNetConfig) -- and A3 becomes exact in the limit
T -> 0+: m(0) is degenerate at 1, the residual smile disappears, the surface is
continuous at expiry, and the short-end sqrt(Var(log m)/T) implied-vol floor of
the fixed-mean design vanishes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT2 = math.sqrt(2.0)


def _ncdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


@dataclass
class DensityNetConfig:
    context_dim: int = 0
    n_components: int = 8
    hidden: List[int] = field(default_factory=lambda: [32, 32])
    # --- Skew clock (v4) ------------------------------------------------------
    # m_i(T) = 1 + h(T) (m_i - 1),  h(T) = 1 - exp(-beta T),  beta = softplus(.) > 0.
    #
    # h is non-decreasing with h(0) = 0 and h in [0, 1), so for T <= T' the
    # mixing law at T is a contraction of the one at T' towards its mean 1:
    #     m(T) = (1 - s) * 1 + s * m(T'),   s = h(T)/h(T') in [0, 1],
    # hence m(T) <=_cx m(T') (Jensen twice), and the two-stage calendar proof
    # (condition on the mixing value, martingale M_T; condition on M_T,
    # convex-order the mixing law) goes through unchanged: (i)-(iii) of the
    # density theorem hold verbatim. E[m(T)] = 1 for every T (martingale),
    # m_i(T) > 0 (convex combination of 1 and m_i > 0), and the price stays a
    # closed-form mixture of Black calls with forwards m_i(T).
    #
    # Payoff vs the fixed-mean design: m(0) is DEGENERATE at 1, so x_{0+} = 1
    # -- the expiry kink is exact, the surface is continuous at T = 0, and
    # short-dated total implied variance loses the additive Var(log m(theta))
    # offset (no sqrt(V_m/T) short-end IV floor); the skew term structure (h)
    # is decoupled from the variance term structure (sigma^2). This is NOT the
    # falsified negative control (per-component e^{mu_i T} means, which broke
    # the x_T = m(theta) M_T factorisation): here the T-dependence is a
    # deterministic dilation about the fixed mean 1, which is exactly what the
    # convex order needs.
    skew_clock: bool = False
    # --- Dispersion-constrained mode (v4 ablation) ----------------------------
    # If set, the mixing law's log-dispersion is CONSTRAINED to
    #     V_m = Var_w(log m_bar) = fixed_V_m
    # by centring and rescaling the raw log-means before the martingale
    # normalisation. This is what makes a dispersion sweep meaningful: a
    # constant skew clock h == s is only a REPARAMETERISATION of h == 1 (the
    # optimiser rescales m_bar to absorb s), so the trade-off of
    # Cor. (cost of a constant skew clock) can only be traced by pinning V_m
    # itself. With fixed_V_m = 0 the mixture collapses to a single lognormal.
    fixed_V_m: Optional[float] = None


class DensityNet(nn.Module):
    """Martingale lognormal-mixture call surface (A1/A2/A4 arbitrage-free by construction)."""

    def __init__(self, config: DensityNetConfig):
        super().__init__()
        self.cfg = config
        M = config.n_components
        self.M = M
        # logits[M], raw_mean[M], sched[2] (+ raw skew-clock rate if enabled)
        out_dim = 2 * M + 2 + (1 if config.skew_clock else 0)
        if config.context_dim > 0:
            dims = [config.context_dim] + list(config.hidden) + [out_dim]
            layers: List[nn.Module] = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(a, b)); layers.append(nn.Tanh())
            layers = layers[:-1]
            self.param_net = nn.Sequential(*layers)
            self.raw = None
            head = self.param_net[-1].bias
        else:
            self.param_net = None
            self.raw = nn.Parameter(torch.zeros(out_dim))
            head = self.raw
        # init: mild mean spread (skew), ATM total var ~0.2^2*T, small curvature
        with torch.no_grad():
            head[M:2 * M] = torch.linspace(-0.3, 0.3, M)        # raw means
            head[2 * M] = math.log(math.exp(0.04) - 1.0)        # softplus -> ~0.04 (linear var)
            head[2 * M + 1] = -5.0                              # softplus -> ~0.0067 (curvature)
            if config.skew_clock:
                # softplus(20) ~ 20: h ramps on a ~2.5-week timescale at init
                # (h(1w) ~ 0.32, h(1m) ~ 0.80), i.e. skew is materially present
                # at monthly maturities but suppressed on the shortest weeklies.
                head[2 * M + 2] = 20.0

    def _head_out(self, context: Optional[torch.Tensor], B: int, device, dtype):
        if self.param_net is not None and context is not None:
            return self.param_net(context)
        return self.raw.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1)

    def _params(self, context: Optional[torch.Tensor], B: int, device, dtype):
        M = self.M
        out = self._head_out(context, B, device, dtype)
        logits, raw_m, sched = out[:, :M], out[:, M:2 * M], out[:, 2 * M:]
        w = torch.softmax(logits, dim=-1)                       # (B, M)
        if self.cfg.fixed_V_m is not None:
            # Pin Var_w(log m_bar) to the target BEFORE the martingale
            # normalisation: centre the raw log-means under w, rescale to the
            # target standard deviation, then let the usual normalisation
            # restore mean 1. The realised Var_w(log m_bar) after
            # normalisation is unchanged, since that step adds a constant to
            # every log-mean.
            mu = (w * raw_m).sum(dim=-1, keepdim=True)
            cen = raw_m - mu
            sd = torch.sqrt((w * cen * cen).sum(dim=-1, keepdim=True).clamp(min=1e-24))
            target = math.sqrt(max(float(self.cfg.fixed_V_m), 0.0))
            raw_m = cen * (target / sd.clamp(min=1e-12))
        am = torch.exp(raw_m)                                   # (B, M) > 0
        m = am / (w * am).sum(dim=-1, keepdim=True).clamp(min=1e-12)   # martingale: sum w_i m_i = 1
        c0 = F.softplus(sched[:, 0:1])                          # (B,1) >= 0
        c1 = F.softplus(sched[:, 1:2])                          # (B,1) >= 0
        return w, m, c0, c1

    def _skew_beta(self, context: Optional[torch.Tensor], B: int, device, dtype):
        """Rate of the skew clock h(T) = 1 - exp(-beta T); None if disabled."""
        if not self.cfg.skew_clock:
            return None
        out = self._head_out(context, B, device, dtype)
        return F.softplus(out[:, 2 * self.M + 2: 2 * self.M + 3])   # (B,1) > 0

    @torch.no_grad()
    def mixture_diagnostics(self, context: Optional[torch.Tensor] = None) -> dict:
        """Mixing-law diagnostics for reporting: V_m = Var(log m(theta)) of the
        FIXED means under the mixture weights (the T->infty mixing law when the
        skew clock is on; the maturity-independent one when it is off), the
        clock coefficients, and -- if the skew clock is enabled -- beta and the
        clock value h at one week. With the skew clock, Var(log m(theta, T))
        shrinks like h(T)^2 V_m for small spreads, so V_m bounds the additive
        short-end total-variance offset the fixed-mean design carries."""
        dev = next(self.parameters()).device
        ctx = None
        B = 1
        if self.param_net is not None and context is not None:
            ctx = context if context.dim() == 2 else context.unsqueeze(0)
            B = ctx.shape[0]
        w, m, c0, c1 = self._params(ctx, B, dev, torch.float32)
        logm = torch.log(m.clamp(min=1e-12))
        mu = (w * logm).sum(-1, keepdim=True)
        V_m = (w * (logm - mu) ** 2).sum(-1)
        out = {"V_m": float(V_m.mean()), "c0": float(c0.mean()), "c1": float(c1.mean())}
        beta = self._skew_beta(ctx, B, dev, torch.float32)
        if beta is not None:
            out["beta"] = float(beta.mean())
            out["h_1w"] = float((1.0 - torch.exp(-beta * (7.0 / 365.25))).mean())
        return out

    def forward(self, K, T, S, r, q=None, context=None) -> dict:
        if q is None:
            q = torch.zeros_like(r)
        B = K.shape[0]
        F_T = S * torch.exp((r - q) * T)
        w, m, c0, c1 = self._params(context, B, K.device, K.dtype)
        T_ = T.unsqueeze(-1)                                    # (B,1)
        if self.cfg.skew_clock:
            beta = self._skew_beta(context, B, K.device, K.dtype)   # (B,1) > 0
            h = 1.0 - torch.exp(-beta * T_.clamp(min=0.0))          # (B,1); h(0)=0, h in [0,1)
            m = 1.0 + h * (m - 1.0)          # (B,M) dilation about 1: sum_i w_i m_i(T) = 1, m_i(T) > 0
        sig2 = c0 * T_ + c1 * T_ * T_                           # (B,1) shared total var, increasing, 0 at T=0
        v = torch.sqrt(sig2.clamp(min=1e-12))                  # (B,1) shared total vol
        kappa = (K / F_T.clamp(min=1e-12)).unsqueeze(-1)       # (B,1) forward-moneyness
        d1 = (torch.log(m / kappa.clamp(min=1e-12)) + 0.5 * v * v) / v
        d2 = d1 - v
        cn = (w * (m * _ncdf(d1) - kappa * _ncdf(d2))).sum(dim=-1)     # (B,) normalized undisc call
        C_fwd = F_T * cn
        C = torch.exp(-r * T) * C_fwd
        intrinsic = torch.exp(-r * T) * torch.clamp(F_T - K, min=0.0)  # exact at expiry
        C = torch.where(T <= 1e-12, intrinsic, C)
        return {"price": C, "C_fwd": C_fwd, "K": K}

    def implied_vol(self, K, T, S, r, q=None, context=None, n_iter: int = 40):
        from .composite import bs_call_price, normal_pdf
        if q is None:
            q = torch.zeros_like(r)
        with torch.no_grad():
            C = self.forward(K, T, S, r, q, context)["price"]
            intrinsic = (S * torch.exp(-q * T) - K * torch.exp(-r * T)).clamp(min=0.0)
            upper = S * torch.exp(-q * T)
            sigma = torch.full_like(C, 0.2)
            valid = (C > intrinsic + 1e-10) & (C < upper - 1e-10)
            T_safe = T.clamp(min=1e-12)
            for _ in range(n_iter):
                bs = bs_call_price(S, K, T, r, sigma, q)
                sig_sqrtT = sigma * torch.sqrt(T_safe)
                d1 = (torch.log((S + 1e-12) / (K + 1e-12)) + (r - q + 0.5 * sigma * sigma) * T_safe) \
                    / sig_sqrtT.clamp(min=1e-12)
                vega = S * torch.exp(-q * T_safe) * normal_pdf(d1) * torch.sqrt(T_safe)
                sigma = (sigma - (bs - C) / vega.clamp(min=1e-12)).clamp(min=1e-6, max=5.0)
            return torch.where(valid, sigma, torch.full_like(sigma, float("nan")))
