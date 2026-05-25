"""Composite training loss for ArbNet and competitor models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .pricing import price_rmse, iv_rmse
from .hedging import hedging_loss


@dataclass
class LossConfig:
    lambda_price: float = 1.0
    lambda_iv: float = 0.5
    lambda_hedge: float = 0.0  # set > 0 to enable joint training
    lambda_arb: float = 0.0    # only used by AckererSoftPenaltyNet
    hedge_metric: str = "variance"
    hedge_confidence: float = 0.95
    price_uses_vega_weight: bool = False


class CompositeLoss(nn.Module):
    """Weighted sum of pricing + hedging + arbitrage penalties."""

    def __init__(self, config: LossConfig):
        super().__init__()
        self.cfg = config

    def forward(
        self,
        pred: dict,
        target_price: torch.Tensor,
        target_iv: Optional[torch.Tensor] = None,
        vega: Optional[torch.Tensor] = None,
        hedge_pnl: Optional[torch.Tensor] = None,
        arb_penalty: Optional[dict] = None,
    ) -> dict:
        terms = {}
        loss = pred["price"].new_zeros(())
        if self.cfg.lambda_price > 0:
            v = vega if self.cfg.price_uses_vega_weight else None
            l = price_rmse(pred["price"], target_price, vega=v)
            terms["price"] = l.detach()
            loss = loss + self.cfg.lambda_price * l
        # IV loss is skipped when (a) lambda_iv = 0, (b) target_iv missing, or
        # (c) the model does not expose 'iv' (e.g., the direct call-price ArbNet
        # exposes IV only via post-hoc Newton inversion, which is non-differentiable).
        if self.cfg.lambda_iv > 0 and target_iv is not None and "iv" in pred and pred["iv"].requires_grad:
            l = iv_rmse(pred["iv"], target_iv)
            terms["iv"] = l.detach()
            loss = loss + self.cfg.lambda_iv * l
        if self.cfg.lambda_hedge > 0 and hedge_pnl is not None:
            l = hedging_loss(hedge_pnl, metric=self.cfg.hedge_metric, confidence=self.cfg.hedge_confidence)
            terms["hedge"] = l.detach()
            loss = loss + self.cfg.lambda_hedge * l
        if self.cfg.lambda_arb > 0 and arb_penalty is not None:
            l = arb_penalty["total"]
            terms["arb"] = l.detach()
            loss = loss + self.cfg.lambda_arb * l
        terms["total"] = loss.detach()
        return {"loss": loss, "components": terms}
