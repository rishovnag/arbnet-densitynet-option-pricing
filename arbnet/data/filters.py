"""Quality filters for raw options data.

Common filters applied before any analysis:
- Drop options with stale prices (zero or missing settle)
- Drop options below intrinsic value or above no-arbitrage upper bound
- Drop options with negligible open interest / volume
- Restrict to a moneyness window |log(K/F)| <= cap
- Restrict to a maturity window
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .loaders import OptionsSnapshot


@dataclass
class FilterConfig:
    min_price: float = 0.05               # below this, treat as untraded
    moneyness_cap: float = 0.5            # |log(K/F)| <= cap
    min_tte_days: int = 1
    max_tte_days: int = 90
    min_oi: float = 0.0
    enforce_no_arb_bounds: bool = True


def apply_quality_filters(snap: OptionsSnapshot, cfg: FilterConfig) -> Tuple[OptionsSnapshot, dict]:
    """Apply filters; return filtered snapshot and a summary of drop counts."""
    n0 = len(snap)
    F = snap.spot * np.exp((snap.risk_free_rate - snap.dividend_yield) * snap.times_to_expiry)
    logm = np.log(snap.strikes / np.maximum(F, 1e-12))

    drop_price = snap.prices < cfg.min_price
    drop_moneyness = np.abs(logm) > cfg.moneyness_cap
    drop_tte_low = snap.times_to_expiry < cfg.min_tte_days / 365.0
    drop_tte_high = snap.times_to_expiry > cfg.max_tte_days / 365.0
    drop_oi = snap.open_interest < cfg.min_oi
    drops = drop_price | drop_moneyness | drop_tte_low | drop_tte_high | drop_oi

    # No-arbitrage upper/lower bounds
    bound_violation = np.zeros(n0, dtype=bool)
    if cfg.enforce_no_arb_bounds:
        # Call bounds: max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}
        S_disc = snap.spot * np.exp(-snap.dividend_yield * snap.times_to_expiry)
        K_disc = snap.strikes * np.exp(-snap.risk_free_rate * snap.times_to_expiry)
        call_lower = np.maximum(S_disc - K_disc, 0.0)
        call_upper = S_disc
        # Put bounds: max(K e^{-rT} - S e^{-qT}, 0) <= P <= K e^{-rT}
        put_lower = np.maximum(K_disc - S_disc, 0.0)
        put_upper = K_disc
        is_call = snap.option_types == "C"
        bound_violation = np.where(
            is_call,
            (snap.prices < call_lower - 1e-6) | (snap.prices > call_upper + 1e-6),
            (snap.prices < put_lower - 1e-6) | (snap.prices > put_upper + 1e-6),
        )
        drops = drops | bound_violation

    keep = ~drops
    filtered = snap._mask(keep)
    summary = {
        "n_initial": int(n0),
        "n_kept": int(keep.sum()),
        "n_dropped_price": int(drop_price.sum()),
        "n_dropped_moneyness": int(drop_moneyness.sum()),
        "n_dropped_tte_low": int(drop_tte_low.sum()),
        "n_dropped_tte_high": int(drop_tte_high.sum()),
        "n_dropped_oi": int(drop_oi.sum()),
        "n_dropped_bounds": int(bound_violation.sum()),
    }
    return filtered, summary
