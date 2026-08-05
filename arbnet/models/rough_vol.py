"""Rough Bergomi spot/variance simulator.

Implementation of the rough Bergomi (Bayer, Friz, Gatheral 2016) variance
process with constant Hurst parameter H and flat forward variance curve.
The fractional driver is built via a Riemann-sum discretization of the
Riemann-Liouville kernel; (W^1, Y) is jointly Gaussian by construction,
so no Cholesky factorization is needed.

Reference:
- Bayer, Friz, Gatheral (2016) "Pricing under rough volatility", Quant. Fin.
- Gatheral, Jaisson, Rosenbaum (2018) "Volatility is rough"
"""
from __future__ import annotations

from typing import Optional

import math
import torch


class RoughBergomiSimulator:
    """Vanilla (non-neural) rough Bergomi simulator for benchmarks.

    Variance process:
        V_t = xi_0(t) * exp(eta * Y_t - 0.5 * eta^2 * E[Y_t^2])
    where Y_t is a Riemann-Liouville fractional Brownian motion with Hurst H,
        Y_t = sqrt(2H) * integral_0^t (t-s)^{H-1/2} dW^1_s.

    Price process:
        dS_t / S_t = (r - q) dt + sqrt(V_t) (rho dW^1_t + sqrt(1-rho^2) dW^2_t).

    Discretisation: a midpoint Riemann sum of the RL kernel (see ``simulate``).
    KNOWN BIAS (H7): the midpoint rule under-counts the singular near-diagonal
    kernel contribution, so Var(Y_T) converges to its theoretical value only as
    O(N^{2H-1}) FROM BELOW -- e.g. ~0.70/0.81 of theory at N=50/500 for H=0.1.
    The effective vol-of-vol is therefore below the nominal ``eta``. The
    simulator remains self-consistent and arbitrage-free as discretised
    (the convexity correction uses the DISCRETISED variance, so E[V_t] = xi0
    and E[S_t] = S0 e^{(r-q)t} hold exactly); only the roughness LEVEL is
    biased. For unbiased small-H simulation switch to the hybrid scheme of
    Bennedsen-Lunde-Pakkanen (2017), which is not implemented here.
    """

    def __init__(
        self,
        H: float = 0.07,
        eta: float = 1.9,
        rho: float = -0.7,
        xi0: float = 0.04,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if not 0.0 < H < 0.5:
            raise ValueError("Rough Bergomi requires 0 < H < 0.5")
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi0 = xi0
        self.device = torch.device(device)
        self.dtype = dtype

    def simulate(self, n_paths: int, t_grid: torch.Tensor, S0: float = 1.0, r: float = 0.0, q: float = 0.0, seed: Optional[int] = None):
        """Simulate (S, V) paths via a Riemann-sum discretization of the RL kernel.

        The Riemann-Liouville fBm  Y_t = sqrt(2H) * integral_0^t (t-u)^{H-1/2} dW_u
        is approximated by
            Y_{t_i} = sqrt(2H) * sum_{j<i} (t_i - t_{j+1/2})^{H-1/2} * dW_j,
        where t_{j+1/2} is the midpoint of [t_j, t_{j+1}]. The midpoint rule
        keeps the singular kernel finite but does NOT remove the leading bias:
        Var(Y_T) is downward-biased with O(N^{2H-1}) convergence (see the
        class docstring, H7). The convexity correction below uses the
        discretised variance, so the simulated measure is a martingale and
        arbitrage-free as discretised; only the roughness level is biased.

        Because Y is a measurable transformation of the same Gaussian increments
        dW_j that build W^1, the joint (W^1, Y) is automatically Gaussian with a
        PSD covariance — no Cholesky factorization is needed.

        Args:
            n_paths: number of Monte Carlo paths.
            t_grid: time grid starting at 0, shape (N+1,).
            S0: spot at t=0.
            r, q: rates.
            seed: optional RNG seed.

        Returns:
            S: shape (n_paths, N+1)
            V: shape (n_paths, N+1)
        """
        if seed is not None:
            g = torch.Generator(device=self.device.type)
            g.manual_seed(seed)
        else:
            g = None

        def _randn(shape):
            if g is not None:
                return torch.randn(*shape, generator=g, device=self.device, dtype=self.dtype)
            return torch.randn(*shape, device=self.device, dtype=self.dtype)

        t_grid = t_grid.to(self.device, self.dtype)
        N = t_grid.shape[0] - 1
        dt = t_grid[1:] - t_grid[:-1]               # (N,)
        sqrt_dt = dt.sqrt()
        midpoints = 0.5 * (t_grid[:-1] + t_grid[1:])  # (N,)

        # Driver increments for W^1
        dW1 = _randn((n_paths, N)) * sqrt_dt.unsqueeze(0)
        # Independent W^2 driver
        dW2 = _randn((n_paths, N)) * sqrt_dt.unsqueeze(0)

        # Lower-triangular kernel matrix: K[i, j] = (t_{i+1} - midpoints[j])^{H-1/2}
        # for j <= i, else 0. Shape (N, N).
        t_eval = t_grid[1:].unsqueeze(1)            # (N, 1) — Y is evaluated at t_1, ..., t_N
        mid = midpoints.unsqueeze(0)                # (1, N)
        gap = (t_eval - mid).clamp(min=1e-12)
        kernel = gap.pow(self.H - 0.5)              # (N, N), but only valid for j <= i
        tril_mask = torch.tril(torch.ones(N, N, dtype=self.dtype, device=self.device))
        kernel = kernel * tril_mask
        # Y_{t_{i+1}} = sqrt(2H) * sum_j kernel[i, j] * dW1[:, j]
        Y_pos = math.sqrt(2.0 * self.H) * dW1 @ kernel.transpose(0, 1)  # (n_paths, N)
        zeros = torch.zeros(n_paths, 1, device=self.device, dtype=self.dtype)
        Y = torch.cat([zeros, Y_pos], dim=1)        # (n_paths, N+1)

        # Variance:  V_t = xi0 * exp(eta * Y_t - 0.5 * eta^2 * Var(Y_t))
        # Theoretical Var(Y_t) = t^{2H}; for the discretization we recompute
        # the row-sums-of-squares of the kernel times 2H * dt for unbiasedness:
        var_Y_disc = 2.0 * self.H * (kernel.pow(2) * dt.unsqueeze(0)).sum(dim=1)   # (N,)
        var_Y = torch.cat([torch.zeros(1, dtype=self.dtype, device=self.device), var_Y_disc], dim=0).unsqueeze(0)
        V = self.xi0 * torch.exp(self.eta * Y - 0.5 * self.eta * self.eta * var_Y)

        # Spot path with correlated Brownian:  d log S = (r - q - V/2) dt + sqrt(V) [rho dW1 + sqrt(1-rho^2) dW2]
        rho2 = math.sqrt(max(1.0 - self.rho * self.rho, 0.0))
        dlogS = ((r - q) - 0.5 * V[:, :-1]) * dt.unsqueeze(0) \
                + V[:, :-1].clamp(min=0.0).sqrt() * (self.rho * dW1 + rho2 * dW2)
        logS = torch.cat([torch.full((n_paths, 1), math.log(S0), device=self.device, dtype=self.dtype),
                          dlogS.cumsum(dim=1) + math.log(S0)], dim=1)
        S = logS.exp()
        return S, V
