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


class DensityNet(nn.Module):
    """Martingale lognormal-mixture call surface (A1/A2/A4 arbitrage-free by construction)."""

    def __init__(self, config: DensityNetConfig):
        super().__init__()
        self.cfg = config
        M = config.n_components
        self.M = M
        out_dim = 2 * M + 2                       # logits[M], raw_mean[M], sched[2]
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

    def _params(self, context: Optional[torch.Tensor], B: int, device, dtype):
        M = self.M
        if self.param_net is not None and context is not None:
            out = self.param_net(context)
        else:
            out = self.raw.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
        logits, raw_m, sched = out[:, :M], out[:, M:2 * M], out[:, 2 * M:]
        w = torch.softmax(logits, dim=-1)                       # (B, M)
        am = torch.exp(raw_m)                                   # (B, M) > 0
        m = am / (w * am).sum(dim=-1, keepdim=True).clamp(min=1e-12)   # martingale: sum w_i m_i = 1
        c0 = F.softplus(sched[:, 0:1])                          # (B,1) >= 0
        c1 = F.softplus(sched[:, 1:2])                          # (B,1) >= 0
        return w, m, c0, c1

    def forward(self, K, T, S, r, q=None, context=None) -> dict:
        if q is None:
            q = torch.zeros_like(r)
        B = K.shape[0]
        F_T = S * torch.exp((r - q) * T)
        w, m, c0, c1 = self._params(context, B, K.device, K.dtype)
        T_ = T.unsqueeze(-1)                                    # (B,1)
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
