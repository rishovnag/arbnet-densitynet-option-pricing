"""Static no-arbitrage checks.

We test three classical conditions on the call price surface C(K, T):

1. Butterfly (strike-convexity): d^2 C / d K^2 >= 0 for each fixed T.
   In total variance space: d^2 (w * something) / dk^2 condition; for simplicity
   we check directly on the price grid.

2. Calendar: e^{q T} C(K, T) non-decreasing in T (under nonneg div yield) at
   fixed forward-moneyness. In total variance: w(k, T) non-decreasing in T at
   each k.

3. Roger Lee tail bound: lim sup |k| -> infinity of (w(k, T) / |k|) <= 2.
   Practical test: estimate the asymptotic slope from the wings of the grid.

These checks are evaluated on a regular grid sampled from the surface. Outputs
are violation counts and continuous "violation magnitude" arrays useful both
for soft-penalty losses and post-hoc evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StaticArbReport:
    butterfly_count: int
    calendar_count: int
    tail_count: int
    butterfly_rate: float
    calendar_rate: float
    tail_rate: float
    total_rate: float

    def __str__(self) -> str:
        return (
            f"Static arbitrage report: "
            f"butterfly {self.butterfly_count} ({self.butterfly_rate:.4%}), "
            f"calendar {self.calendar_count} ({self.calendar_rate:.4%}), "
            f"tail {self.tail_count} ({self.tail_rate:.4%}), "
            f"total {self.total_rate:.4%}"
        )


def butterfly_violations(C: torch.Tensor, K: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Detect butterfly arbitrage on a price grid.

    Tests d^2 C / d K^2 >= 0 via second differences.

    Args:
        C: call prices on grid, shape (..., n_K).
        K: strikes, shape (..., n_K) or (n_K,). Must be sorted ascending along dim.
        dim: strike dimension.

    Returns:
        Tensor of violation magnitudes (>= 0; zero means no violation), shape
        (..., n_K - 2). A positive value indicates negative second difference
        (curvature violation).
    """
    if K.dim() != C.dim():
        K = K.expand_as(C)
    C_l = C.narrow(dim, 0, C.size(dim) - 2)
    C_m = C.narrow(dim, 1, C.size(dim) - 2)
    C_r = C.narrow(dim, 2, C.size(dim) - 2)
    K_l = K.narrow(dim, 0, K.size(dim) - 2)
    K_m = K.narrow(dim, 1, K.size(dim) - 2)
    K_r = K.narrow(dim, 2, K.size(dim) - 2)
    # Three-point convexity test: weighted second difference. For non-uniform K,
    # use the divided-difference formula:
    #   D2 = ( (C_r - C_m)/(K_r - K_m) - (C_m - C_l)/(K_m - K_l) ) / ((K_r - K_l)/2)
    h1 = K_m - K_l
    h2 = K_r - K_m
    slope_lo = (C_m - C_l) / h1.clamp(min=1e-12)
    slope_hi = (C_r - C_m) / h2.clamp(min=1e-12)
    second = (slope_hi - slope_lo) / ((h1 + h2) / 2.0).clamp(min=1e-12)
    # Violation magnitude: -second clipped to nonneg
    return torch.relu(-second)


def calendar_violations(w: torch.Tensor, T: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Detect calendar arbitrage in total variance.

    Tests w(k, T) non-decreasing in T at each k.

    Args:
        w: total variance on grid, shape (..., n_T).
        T: maturities, shape (..., n_T) or (n_T,). Sorted ascending.
        dim: time dimension.

    Returns:
        Violation magnitudes, shape (..., n_T - 1). Positive means decrease.
    """
    w_lo = w.narrow(dim, 0, w.size(dim) - 1)
    w_hi = w.narrow(dim, 1, w.size(dim) - 1)
    return torch.relu(w_lo - w_hi)


def calendar_violations_price(
    C: torch.Tensor, T: torch.Tensor, r: float, q: float = 0.0, dim: int = -1
) -> torch.Tensor:
    """Detect calendar arbitrage directly on call prices.

    For q = 0, the no-calendar-arbitrage condition is that e^{rT} * C(K, T)
    is non-decreasing in T at fixed strike K. For q > 0 with r >= q (typical),
    the same direction of inequality holds for max(F_T - K, 0) so we use the
    carry-adjusted price e^{(r - q)T} * C as the surrogate, which reduces to
    e^{rT} * C when q = 0.

    Args:
        C: call prices on grid, last axis is K, second-to-last is T. Shape
           (..., n_T, n_K) or (..., n_T).
        T: maturities, shape (n_T,).
        r: risk-free rate.
        q: dividend yield (default 0).
        dim: time dimension to check along.

    Returns:
        Violation magnitudes (>= 0; zero means no violation), same shape as
        C with size along dim reduced by 1.
    """
    carry = torch.exp(torch.tensor(float(r - q)) * T)
    # Broadcast carry across the K axis: assume dim is the T-axis and the last
    # axis of C corresponds to K.
    if dim == -1 or dim == C.dim() - 1:
        carry_b = carry
    else:
        # Reshape carry to broadcast over later dims
        view = [1] * C.dim()
        view[dim] = T.shape[0]
        carry_b = carry.view(view)
    Cw = C * carry_b
    lo = Cw.narrow(dim, 0, Cw.size(dim) - 1)
    hi = Cw.narrow(dim, 1, Cw.size(dim) - 1)
    return torch.relu(lo - hi)


def roger_lee_tail_violations(
    w: torch.Tensor,
    k: torch.Tensor,
    tail_fraction: float = 0.2,
    bound: float = 2.0,
    dim: int = -1,
) -> torch.Tensor:
    """Detect Roger Lee tail-bound violations.

    Estimates the empirical wing slope of w as a function of |k| from the
    extreme tail_fraction of the grid (each side) and checks whether it
    exceeds ``bound`` (the no-arbitrage bound from Lee 2004, equal to 2).

    Args:
        w: total variance, shape (..., n_k).
        k: log-moneyness grid, shape (..., n_k) or (n_k,). Sorted ascending.
        tail_fraction: fraction of grid points to use on each wing.
        bound: maximum allowed asymptotic slope of w wrt |k| (default 2).
        dim: strike dimension.

    Returns:
        Tensor of shape (..., 2) with violation magnitudes for (left wing, right wing).
    """
    if k.dim() != w.dim():
        k = k.expand_as(w)
    n = w.size(dim)
    n_tail = max(2, int(n * tail_fraction))
    # Right tail
    w_right = w.narrow(dim, n - n_tail, n_tail)
    k_right = k.narrow(dim, n - n_tail, n_tail)
    # Least-squares slope of w vs k on the right tail
    slope_r = _linreg_slope(k_right, w_right, dim=dim)
    # Left tail (slope vs -k)
    w_left = w.narrow(dim, 0, n_tail)
    k_left = k.narrow(dim, 0, n_tail)
    slope_l = -_linreg_slope(k_left, w_left, dim=dim)
    # Violations are excess over `bound`
    viol_r = torch.relu(slope_r - bound)
    viol_l = torch.relu(slope_l - bound)
    return torch.stack([viol_l, viol_r], dim=-1)


def _linreg_slope(x: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    """Slope of OLS regression of y on x along dim."""
    n = x.size(dim)
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    num = ((x - x_mean) * (y - y_mean)).sum(dim=dim)
    den = ((x - x_mean) ** 2).sum(dim=dim).clamp(min=1e-12)
    return num / den


def static_arbitrage_report(
    C_grid: torch.Tensor,
    w_grid: torch.Tensor,
    K_grid: torch.Tensor,
    T_grid: torch.Tensor,
    k_grid: torch.Tensor,
    r: float = 0.0,
    q: float = 0.0,
    tail_fraction: float = 0.2,
    tol: float = 1e-8,
) -> StaticArbReport:
    """Aggregate all three checks over a (T, K)-grid into a report.

    The calendar check is performed in the price domain (e^{(r-q)T} C
    non-decreasing in T at fixed K), which is the universally correct condition
    valid for arbitrary q. The tail check is performed in implied-total-variance
    space via Roger Lee's bound.

    Args:
        C_grid: call prices, shape (n_T, n_K).
        w_grid: total variance, shape (n_T, n_K).
        K_grid: strikes, shape (n_K,) or (n_T, n_K).
        T_grid: maturities, shape (n_T,) or (n_T, n_K).
        k_grid: log-moneyness, shape (n_K,) or (n_T, n_K).
        r: risk-free rate used for the calendar carry adjustment.
        q: dividend yield (default 0).
        tail_fraction: tail fraction for the Lee check.
        tol: tolerance below which violations are not counted.
    """
    if K_grid.dim() == 1:
        K_grid_b = K_grid.unsqueeze(0).expand_as(C_grid)
    else:
        K_grid_b = K_grid
    # Tolerance: absolute floor 1e-7 (one decade above f32 machine epsilon)
    # is appropriate for second-difference magnitudes which inherit float noise
    # of order eps * price / (typical strike spacing). For Nifty prices ~ 100s
    # and strike spacing ~ 500, eps_f32 * price / dK^2 ~ 1e-7 * 100 / 250000 ~ 4e-11,
    # so 1e-7 is comfortably above noise floor without missing economic violations
    # (which are typically O(1e-4) or larger in our experiments).
    eff_tol = max(tol, 1e-7)
    bf = butterfly_violations(C_grid, K_grid_b, dim=-1)
    bf_count = int((bf > eff_tol).sum().item())
    bf_total = int(bf.numel())

    # Calendar: e^{(r-q)T} * C non-decreasing in T at fixed K (column-wise).
    cal = calendar_violations_price(C_grid, T_grid, r=r, q=q, dim=0)
    cal_count = int((cal > eff_tol).sum().item())
    cal_total = int(cal.numel())

    if k_grid.dim() == 1:
        k_grid_b = k_grid.unsqueeze(0).expand_as(w_grid)
    else:
        k_grid_b = k_grid
    tail = roger_lee_tail_violations(w_grid, k_grid_b, tail_fraction=tail_fraction, dim=-1)
    tail_count = int((tail > tol).sum().item())
    tail_total = int(tail.numel())

    total_count = bf_count + cal_count + tail_count
    total_total = bf_total + cal_total + tail_total
    return StaticArbReport(
        butterfly_count=bf_count,
        calendar_count=cal_count,
        tail_count=tail_count,
        butterfly_rate=bf_count / max(1, bf_total),
        calendar_rate=cal_count / max(1, cal_total),
        tail_rate=tail_count / max(1, tail_total),
        total_rate=total_count / max(1, total_total),
    )
