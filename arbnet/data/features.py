"""Feature engineering helpers."""
from __future__ import annotations

import numpy as np
import torch

from .loaders import OptionsSnapshot


def log_moneyness(K: np.ndarray, S: float, T: np.ndarray, r: float, q: float) -> np.ndarray:
    """Forward log-moneyness k = log(K / F), F = S exp((r-q) T)."""
    F = S * np.exp((r - q) * T)
    return np.log(K / np.maximum(F, 1e-12))


def build_features(snap: OptionsSnapshot, context: dict | None = None) -> dict:
    """Build tensors for model input from a snapshot.

    Args:
        snap: filtered OptionsSnapshot.
        context: optional dict of auxiliary scalar features (e.g. 'india_vix',
            'fii_dii', etc.). Each becomes one column of the context tensor.

    Returns:
        Dict of torch tensors: 'k', 'T', 'S', 'r', 'q', 'price', 'iv', and
        optionally 'context'.
    """
    k_np = log_moneyness(snap.strikes, snap.spot, snap.times_to_expiry, snap.risk_free_rate, snap.dividend_yield)
    n = len(snap)
    out = {
        "k": torch.tensor(k_np, dtype=torch.float32),
        "T": torch.tensor(snap.times_to_expiry, dtype=torch.float32),
        "S": torch.full((n,), float(snap.spot), dtype=torch.float32),
        "r": torch.full((n,), float(snap.risk_free_rate), dtype=torch.float32),
        "q": torch.full((n,), float(snap.dividend_yield), dtype=torch.float32),
        "price": torch.tensor(snap.prices, dtype=torch.float32),
        "K": torch.tensor(snap.strikes, dtype=torch.float32),
    }
    if snap.implied_vol is not None:
        out["iv"] = torch.tensor(snap.implied_vol, dtype=torch.float32)
    if context is not None and len(context) > 0:
        ctx_arr = np.stack([np.full(n, float(v), dtype=np.float32) for v in context.values()], axis=-1)
        out["context"] = torch.tensor(ctx_arr, dtype=torch.float32)
    return out
