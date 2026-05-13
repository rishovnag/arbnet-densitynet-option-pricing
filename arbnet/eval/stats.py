"""Statistical tests for model comparison.

- bootstrap_ci: percentile bootstrap confidence intervals.
- diebold_mariano: DM test for equal forecast accuracy (Diebold & Mariano 1995),
  with Harvey-Leybourne-Newbold (1997) small-sample correction.
- paired_permutation_test: paired sign-flip permutation test.
- holm_bonferroni: Holm's step-down multiple-comparison correction.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Callable, List

import math
import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a statistic.

    Returns (point_estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = statistic(data[idx])
    alpha = (1.0 - confidence) / 2.0
    return float(statistic(data)), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha))


def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss: str = "squared",
) -> dict:
    """Diebold-Mariano test of equal forecast accuracy with HLN correction.

    Args:
        e1, e2: per-observation forecast errors of model 1 and model 2.
        h: forecast horizon (used to set HAC lag = h - 1).
        loss: 'squared' or 'absolute'.

    Returns:
        Dict with 'dm_stat', 'p_value', 'mean_loss_diff'.
        Negative dm_stat means model 1 has lower loss.
    """
    e1, e2 = np.asarray(e1), np.asarray(e2)
    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")
    n = len(d)
    d_bar = d.mean()
    # Newey-West HAC variance with lag = h - 1
    lag = max(0, h - 1)
    gamma0 = ((d - d_bar) ** 2).mean()
    var = gamma0
    for k in range(1, lag + 1):
        gamma_k = ((d[k:] - d_bar) * (d[:-k] - d_bar)).mean()
        weight = 1.0 - k / (lag + 1)
        var += 2.0 * weight * gamma_k
    var = max(var, 1e-12)
    dm = d_bar / math.sqrt(var / n)
    # HLN small-sample correction
    correction = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    # Use t-distribution with n-1 df
    # (Approx via standard normal here; for small n one should use scipy.stats.t)
    from math import erf
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm_hln) / math.sqrt(2.0))))
    return {"dm_stat": float(dm_hln), "p_value": float(p), "mean_loss_diff": float(d_bar)}


def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 0,
) -> dict:
    """Paired permutation (sign-flip) test for H0: x and y have equal means.

    Returns dict with 'observed', 'p_value'.
    """
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    d = x - y
    obs = statistic(d)
    n = len(d)
    more_extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        if abs(statistic(d * signs)) >= abs(obs):
            more_extreme += 1
    p = (more_extreme + 1) / (n_perm + 1)
    return {"observed": float(obs), "p_value": float(p)}


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm step-down procedure.

    Returns a list of booleans indicating which p-values are rejected at FWER alpha.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    for k, idx in enumerate(order):
        threshold = alpha / (m - k)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected
