"""Heston calibration via gradient descent on price RMSE."""
from __future__ import annotations

import time

import torch

from ..models.baselines import HestonPricer


def calibrate_heston(
    features: dict,
    n_steps: int = 500,
    lr: float = 5e-2,
    verbose: bool = False,
) -> tuple[HestonPricer, dict]:
    """Calibrate a Heston model to a snapshot using Adam.

    Args:
        features: dict with 'k', 'T', 'S', 'r', 'q', 'price'.
        n_steps: number of optimization steps.
        lr: Adam learning rate.

    Returns:
        Fitted HestonPricer and a dict with calibration diagnostics.
    """
    model = HestonPricer()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    history = []
    for step in range(n_steps):
        opt.zero_grad()
        out = model(features["k"], features["T"], features["S"], features["r"], features.get("q"))
        loss = (out["price"] - features["price"]).pow(2).mean()
        loss.backward()
        # Heston grads can blow up early on; clip them
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()
        history.append(float(loss.item()))
        if verbose and step % max(1, n_steps // 10) == 0:
            print(f"[Heston] step {step}: price MSE = {loss.item():.6f}")
    return model, {"loss_history": history, "wall_time": time.time() - t0}
