"""Configuration for an experimental run."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List

from ..models.composite import ArbNetConfig


@dataclass
class RunConfig:
    seed: int = 0
    n_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 5.0
    eval_every: int = 25
    device: str = "cpu"
    arbnet: ArbNetConfig = field(default_factory=ArbNetConfig)
    # Loss weights
    lambda_price: float = 1.0
    lambda_iv: float = 0.5
    lambda_hedge: float = 0.0
    lambda_arb: float = 1.0  # only used for the Ackerer baseline
    hedge_metric: str = "variance"

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


def default_arbnet_config(context_dim: int = 0) -> ArbNetConfig:
    return ArbNetConfig(
        context_dim=context_dim,
        n_experts=6,
        icnn_hidden=[64, 64],
        monotone_hidden=[32, 32],
        envelope_hidden=32,
        slope_cap=1.8,
        activation="softplus",
    )
