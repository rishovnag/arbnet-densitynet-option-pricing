"""Synthetic option-surface generators.

Two generators are provided:

1. RoughBergomiGenerator: simulates spot paths and prices a vanilla call grid via
   Monte Carlo. Produces a realistic but expensive ground-truth surface.

2. SyntheticSurfaceGenerator: parametric SSVI / SABR-like surface for fast
   smoke tests. Cheap and arbitrage-free by construction.

Both produce OptionsSnapshot objects compatible with the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import math
import numpy as np
import pandas as pd
import torch

from ..models.rough_vol import RoughBergomiSimulator
from ..models.svi_envelope import ssvi_total_variance
from .loaders import OptionsSnapshot


@dataclass
class SyntheticSurfaceGenerator:
    """Fast SSVI-based generator. Arbitrage-free by construction."""

    base_theta: float = 0.04           # ATM total variance per unit T
    base_rho: float = -0.5
    base_phi: float = 1.0
    spot: float = 20000.0              # roughly Nifty levels
    r: float = 0.065
    q: float = 0.012
    seed: int = 0

    def generate(
        self,
        n_strikes: int = 21,
        moneyness_range: tuple = (-0.3, 0.3),
        maturities_days: List[int] = (7, 14, 30, 60, 90),
        snapshot_date: Optional[pd.Timestamp] = None,
    ) -> OptionsSnapshot:
        rng = np.random.default_rng(self.seed)
        k_arr = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)
        T_arr = np.array(maturities_days, dtype=float) / 365.0
        all_strikes, all_expiries, all_T, all_price, all_iv, all_type = [], [], [], [], [], []
        snap_date = snapshot_date or pd.Timestamp.today().normalize()
        for T in T_arr:
            F = self.spot * np.exp((self.r - self.q) * T)
            theta = self.base_theta * T  # total variance ~ theta * T
            for k in k_arr:
                k_t = torch.tensor(float(k))
                T_t = torch.tensor(float(T))
                w = ssvi_total_variance(k_t, torch.tensor(self.base_theta * T), torch.tensor(self.base_rho), torch.tensor(self.base_phi)).item()
                iv = math.sqrt(max(w / T, 1e-8))
                K = F * math.exp(k)
                # Add small log-normal noise to mimic market microstructure
                price = _bs_call_np(self.spot, K, T, self.r, iv, self.q)
                price *= rng.lognormal(0.0, 0.005)
                all_strikes.append(K)
                all_expiries.append(snap_date + pd.Timedelta(days=int(T * 365)))
                all_T.append(T)
                all_price.append(price)
                all_iv.append(iv)
                all_type.append("C")
        return OptionsSnapshot(
            snapshot_date=snap_date,
            underlying="SYNTH",
            spot=float(self.spot),
            risk_free_rate=self.r,
            dividend_yield=self.q,
            strikes=np.array(all_strikes),
            expiries=np.array([np.datetime64(e) for e in all_expiries]),
            times_to_expiry=np.array(all_T),
            option_types=np.array(all_type),
            prices=np.array(all_price),
            open_interest=np.full(len(all_strikes), 1000.0),
            implied_vol=np.array(all_iv),
        )


@dataclass
class RoughBergomiGenerator:
    """Slow but realistic: rough Bergomi Monte Carlo pricing."""

    H: float = 0.07
    eta: float = 1.9
    rho: float = -0.7
    xi0: float = 0.04
    spot: float = 20000.0
    r: float = 0.065
    q: float = 0.012
    n_paths: int = 4000
    seed: int = 0

    def generate(
        self,
        n_strikes: int = 21,
        moneyness_range: tuple = (-0.3, 0.3),
        maturities_days: List[int] = (7, 14, 30, 60, 90),
        snapshot_date: Optional[pd.Timestamp] = None,
    ) -> OptionsSnapshot:
        snap_date = snapshot_date or pd.Timestamp.today().normalize()
        T_arr = np.array(maturities_days, dtype=float) / 365.0
        T_max = T_arr.max()
        # Build a fine time grid up to T_max
        n_steps = max(50, int(T_max * 365))
        t_grid = torch.linspace(0.0, float(T_max), n_steps + 1, dtype=torch.float64)
        sim = RoughBergomiSimulator(H=self.H, eta=self.eta, rho=self.rho, xi0=self.xi0, dtype=torch.float64)
        S_paths, V_paths = sim.simulate(self.n_paths, t_grid, S0=self.spot, r=self.r, q=self.q, seed=self.seed)
        # Collect prices at each maturity
        all_strikes, all_expiries, all_T, all_price, all_iv, all_type = [], [], [], [], [], []
        k_arr = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)
        for T in T_arr:
            idx = int(round(float(T) / float(T_max) * n_steps))
            S_T = S_paths[:, idx]
            F_T = self.spot * math.exp((self.r - self.q) * float(T))
            for k in k_arr:
                K = F_T * math.exp(float(k))
                payoff = (S_T - K).clamp(min=0.0).double()
                disc = math.exp(-self.r * float(T))
                price = float(disc * payoff.mean().item())
                # Invert IV (cheap Newton)
                iv = _implied_vol_newton_scalar(price, self.spot, K, float(T), self.r, self.q)
                all_strikes.append(K)
                all_expiries.append(snap_date + pd.Timedelta(days=int(T * 365)))
                all_T.append(float(T))
                all_price.append(price)
                all_iv.append(iv)
                all_type.append("C")
        return OptionsSnapshot(
            snapshot_date=snap_date,
            underlying="SYNTH_RBERGOMI",
            spot=float(self.spot),
            risk_free_rate=self.r,
            dividend_yield=self.q,
            strikes=np.array(all_strikes),
            expiries=np.array([np.datetime64(e) for e in all_expiries]),
            times_to_expiry=np.array(all_T),
            option_types=np.array(all_type),
            prices=np.array(all_price),
            open_interest=np.full(len(all_strikes), 1000.0),
            implied_vol=np.array(all_iv),
        )


# --- Numpy helpers ---

def _norm_cdf_np(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call_np(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    sig_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sig_sqrt_T
    d2 = d1 - sig_sqrt_T
    return S * math.exp(-q * T) * _norm_cdf_np(d1) - K * math.exp(-r * T) * _norm_cdf_np(d2)


def _implied_vol_newton_scalar(C, S, K, T, r, q=0.0, tol=1e-7, max_iter=80):
    """Scalar Newton inverter for BS call IV. Returns NaN if non-convergent."""
    intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    upper = S * math.exp(-q * T)
    if C <= intrinsic + 1e-10:
        return 1e-6
    if C >= upper - 1e-10:
        return float("nan")
    sigma = 0.2
    for _ in range(max_iter):
        price = _bs_call_np(S, K, T, r, sigma, q)
        # vega
        if T <= 0:
            return float("nan")
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * math.exp(-q * T) * math.sqrt(T) * (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1**2)
        if vega < 1e-12:
            return float("nan")
        diff = price - C
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        if sigma < 1e-6:
            sigma = 1e-6
        if sigma > 5.0:
            sigma = 5.0
    return sigma
