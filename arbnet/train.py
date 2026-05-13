"""Main training loop for ArbNet and competitor models.

Optimization is Adam on a price-RMSE objective with optional IV-RMSE,
hedging tail-risk, and soft no-arbitrage penalty terms. ArbNet itself does
not require the soft penalty (it is arbitrage-free by construction) — that
term is used only for the Ackerer-style soft-penalty baseline.
"""
from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn

from .losses import CompositeLoss, LossConfig
from .losses.pricing import soft_no_arbitrage_penalty
from .utils.config import RunConfig
from .utils.seed import set_seed


def _select_batch(feats: dict, idx: torch.Tensor) -> dict:
    n_ref = feats["K"].size(0)
    out = {}
    for k, v in feats.items():
        if torch.is_tensor(v) and v.size(0) == n_ref:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def _call_model(model: nn.Module, batch: dict) -> dict:
    return model(
        batch["K"], batch["T"], batch["S"], batch["r"],
        batch.get("q"), batch.get("context"),
    )


def train_pricer(
    model: nn.Module,
    features: dict,
    cfg: RunConfig,
    soft_penalty_grid: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """Train a pricer model on a snapshot.

    Args:
        model: a pricer with forward(K, T, S, r, q, context) -> dict with 'price'.
        features: dict with tensors 'K', 'T', 'S', 'r', 'q', 'price', and
            optionally 'iv' and 'context'.
        cfg: RunConfig.
        soft_penalty_grid: optional dict {'K_grid', 'T_grid'} for the soft
            no-arbitrage penalty (used only when cfg.lambda_arb > 0).
        verbose: print progress.

    Returns:
        Diagnostics dict {'loss_history', 'wall_time'}.
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    for k, v in features.items():
        if torch.is_tensor(v):
            features[k] = v.to(device)

    loss_cfg = LossConfig(
        lambda_price=cfg.lambda_price,
        lambda_iv=cfg.lambda_iv,
        lambda_hedge=cfg.lambda_hedge,
        lambda_arb=cfg.lambda_arb,
        hedge_metric=cfg.hedge_metric,
    )
    loss_fn = CompositeLoss(loss_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = features["K"].size(0)

    history = []
    t0 = time.time()
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            batch = _select_batch(features, idx)
            opt.zero_grad()
            pred = _call_model(model, batch)
            arb_pen = None
            if cfg.lambda_arb > 0 and soft_penalty_grid is not None:
                arb_pen = soft_no_arbitrage_penalty(
                    model,
                    soft_penalty_grid["K_grid"].to(device),
                    soft_penalty_grid["T_grid"].to(device),
                    S=batch["S"][:1],
                    r=batch["r"][:1],
                    q=batch.get("q", torch.zeros_like(batch["r"][:1]))[:1] if "q" in batch else None,
                )
            target_iv = batch["iv"] if "iv" in batch else None
            out = loss_fn(
                pred=pred,
                target_price=batch["price"],
                target_iv=target_iv,
                arb_penalty=arb_pen,
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_losses.append(float(out["loss"].item()))
        avg = sum(epoch_losses) / max(1, len(epoch_losses))
        history.append(avg)
        if verbose and (epoch % cfg.eval_every == 0 or epoch == cfg.n_epochs - 1):
            print(f"[train] epoch {epoch}/{cfg.n_epochs}  loss={avg:.6f}")
    return {"loss_history": history, "wall_time": time.time() - t0}
