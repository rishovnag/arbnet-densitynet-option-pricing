"""Evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def rmse(pred, target) -> float:
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mape(pred, target, eps: float = 1e-6) -> float:
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    return float(np.mean(np.abs((pred - target) / (np.abs(target) + eps))))


def mean_abs_error(pred, target) -> float:
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    return float(np.mean(np.abs(pred - target)))


@dataclass
class HedgePnLStats:
    mean: float
    std: float
    semidev: float
    cvar_95: float
    cvar_99: float
    sharpe: float


def hedge_pnl_stats(pnl) -> HedgePnLStats:
    if torch.is_tensor(pnl):
        pnl = pnl.detach().cpu().numpy()
    pnl = np.asarray(pnl, dtype=float)
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1))
    semidev = float(np.sqrt(np.mean(np.minimum(pnl, 0.0) ** 2)))
    losses = -pnl
    sorted_losses = np.sort(losses)[::-1]
    k95 = max(1, int(0.05 * len(sorted_losses)))
    k99 = max(1, int(0.01 * len(sorted_losses)))
    cvar95 = float(sorted_losses[:k95].mean())
    cvar99 = float(sorted_losses[:k99].mean())
    sharpe = float(mean / std) if std > 1e-12 else float("nan")
    return HedgePnLStats(mean=mean, std=std, semidev=semidev, cvar_95=cvar95, cvar_99=cvar99, sharpe=sharpe)
