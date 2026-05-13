"""Monotone neural network.

Implementation following Daniels & Velikova (2010) and Sill (1998):
a feedforward network where the output is monotone non-decreasing in a
designated subset of input coordinates. Monotonicity is achieved by
constraining the relevant weights to be non-negative and using a monotone
non-decreasing activation function.

We use this to enforce calendar arbitrage absence: total implied variance
w(k, T) = sigma_IV^2(k, T) * T must be non-decreasing in T for each k.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pos(w: torch.Tensor) -> torch.Tensor:
    return F.softplus(w)


class MonotoneLayer(nn.Module):
    """Linear layer where columns in ``monotone_cols`` have non-negative weights."""

    def __init__(self, in_dim: int, out_dim: int, monotone_cols: Sequence[int]):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.monotone_cols = list(monotone_cols)
        # Free part: columns that are NOT monotone
        free_cols = [i for i in range(in_dim) if i not in self.monotone_cols]
        self.free_cols = free_cols
        if len(free_cols) > 0:
            self.W_free = nn.Parameter(torch.randn(out_dim, len(free_cols)) * 0.1)
        else:
            self.register_parameter("W_free", None)
        if len(self.monotone_cols) > 0:
            # Initialize raw weights so that softplus(W_mono) ~= 1/sqrt(in_dim) on average,
            # matching standard-init scale and preventing exponential growth through layers.
            import math as _math
            target = 1.0 / max(_math.sqrt(max(in_dim, 1)), 1.0)
            mean_raw = _math.log(_math.exp(target) - 1.0) if target < 5 else target
            self.W_mono = nn.Parameter(torch.randn(out_dim, len(self.monotone_cols)) * 0.1 + mean_raw)
        else:
            self.register_parameter("W_mono", None)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reassemble the full effective weight matrix
        W_eff = torch.zeros(self.out_dim, self.in_dim, device=x.device, dtype=x.dtype)
        if self.W_free is not None:
            for j_local, j_global in enumerate(self.free_cols):
                W_eff[:, j_global] = self.W_free[:, j_local]
        if self.W_mono is not None:
            W_mono_pos = _pos(self.W_mono)
            for j_local, j_global in enumerate(self.monotone_cols):
                W_eff[:, j_global] = W_mono_pos[:, j_local]
        return F.linear(x, W_eff, self.b)


class MonotoneNet(nn.Module):
    """Feedforward network monotone non-decreasing in selected input columns.

    Uses softplus activations (which are non-decreasing) and non-negative weights
    in the columns corresponding to monotone inputs.

    Args:
        input_dim: Total input dimensionality.
        hidden_dims: Hidden layer sizes.
        output_dim: Output size (default 1).
        monotone_cols: Indices of input columns in which output must be monotone
            non-decreasing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        monotone_cols: Sequence[int] = (0,),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.monotone_cols = list(monotone_cols)

        layers: List[nn.Module] = []
        prev = input_dim
        # First layer respects monotone_cols on the original inputs
        current_monotone_cols = self.monotone_cols
        for h in hidden_dims:
            layers.append(MonotoneLayer(prev, h, current_monotone_cols))
            prev = h
            # Internal hidden activations are all non-negative under softplus, so
            # subsequent layers should treat ALL columns as monotone to preserve
            # global monotonicity (this is the standard Sill construction).
            current_monotone_cols = list(range(h))
        layers.append(MonotoneLayer(prev, output_dim, list(range(prev))))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.softplus(x)
        return x
