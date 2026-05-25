"""Input-Convex Neural Network (ICNN).

Implementation of Amos, Xu, Kolter (ICML 2017) "Input Convex Neural Networks".

We use a Fully Input Convex Neural Network (FICNN) variant:
    z_{i+1} = activation(W_i^{(z)} z_i + W_i^{(y)} y + b_i)
with W^{(z)} constrained to be non-negative (enforced via softplus or clamp)
and activation chosen to be convex non-decreasing (ReLU, softplus).

This guarantees the output is convex in y. In our pricing context, y is the
log-strike (or strike) dimension, ensuring butterfly arbitrage cannot occur
in the strike direction.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive(w: torch.Tensor) -> torch.Tensor:
    """Map a real weight to a non-negative weight via softplus.

    Softplus is preferred to clamp(min=0) because it gives nonzero gradients
    when the underlying parameter goes negative; this is well-known to be
    important for training stability in ICNNs (Amos et al., §3.3).
    """
    return F.softplus(w)


class ICNNLayer(nn.Module):
    """One layer of an ICNN.

    Two affine maps:
        z_path: from previous hidden state z (must use non-negative weights)
        y_path: from input y (unconstrained)
    """

    def __init__(self, in_z: int, in_y: int, out_dim: int, use_z: bool = True):
        super().__init__()
        self.use_z = use_z
        if use_z:
            # We want softplus(W_z) to be O(1/sqrt(in_z)), giving standard-style
            # init magnitudes. softplus(x) ~= 1/sqrt(in_z)  =>  x ~= log(exp(1/sqrt(in_z)) - 1).
            # For in_z = 64, target softplus ~= 0.125, so raw weights ~= -1.99.
            import math as _math
            target = 1.0 / max(_math.sqrt(max(in_z, 1)), 1.0)
            mean_raw = _math.log(_math.exp(target) - 1.0) if target < 5 else target
            self.W_z = nn.Parameter(torch.randn(out_dim, in_z) * 0.1 + mean_raw)
        else:
            self.register_parameter("W_z", None)
        self.W_y = nn.Linear(in_y, out_dim)

    def forward(self, z: Optional[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        out = self.W_y(y)
        if self.use_z and z is not None:
            W_z_pos = _positive(self.W_z)
            out = out + F.linear(z, W_z_pos)
        return out


class ICNN(nn.Module):
    """Fully Input Convex Neural Network.

    Output is convex in the input ``y``. Convex non-decreasing activations
    (softplus) are used internally; the last layer is linear with non-negative
    weights from the hidden state (so it is also convex in y).

    Args:
        input_dim: Dimensionality of the input y in which output is convex.
        hidden_dims: List of hidden layer widths.
        output_dim: Output dimensionality (default 1 for scalar price).
        context_dim: Dimensionality of auxiliary context (concatenated to y as
            extra unconstrained features; convexity is only in the first
            ``input_dim`` coordinates).
        activation: Convex non-decreasing activation; 'softplus' or 'relu'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        context_dim: int = 0,
        activation: str = "softplus",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.activation = activation

        # The first ICNN layer has no z (the previous hidden state); subsequent ones do.
        layers: List[ICNNLayer] = []
        prev = 0
        y_dim = input_dim + context_dim
        for i, h in enumerate(hidden_dims):
            layers.append(ICNNLayer(in_z=prev, in_y=y_dim, out_dim=h, use_z=(i > 0)))
            prev = h
        # Final layer: scalar (or output_dim) linear in z with positive weights, plus y-path
        layers.append(ICNNLayer(in_z=prev, in_y=y_dim, out_dim=output_dim, use_z=True))
        self.layers = nn.ModuleList(layers)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "softplus":
            return F.softplus(x)
        if self.activation == "relu":
            return F.relu(x)
        raise ValueError(f"Unknown activation {self.activation}")

    def forward(self, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the (convex-in-y) output.

        Args:
            y: Tensor of shape (B, input_dim) — the variable of convexity.
            context: Optional tensor of shape (B, context_dim).

        Returns:
            Tensor of shape (B, output_dim).
        """
        if context is not None and self.context_dim > 0:
            y_full = torch.cat([y, context], dim=-1)
        else:
            y_full = y
        z: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.layers):
            pre = layer(z, y_full)
            if i < len(self.layers) - 1:
                z = self._act(pre)
            else:
                z = pre  # linear output
        return z

    def hessian_diag(self, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Approximate diagonal of the Hessian wrt y using autograd.

        Useful as a sanity check that convexity holds (diagonal should be >= 0).
        """
        y = y.clone().detach().requires_grad_(True)
        out = self.forward(y, context).sum()
        grad = torch.autograd.grad(out, y, create_graph=True)[0]
        diag = []
        for j in range(y.shape[-1]):
            g_j = grad[..., j].sum()
            h_j = torch.autograd.grad(g_j, y, retain_graph=True)[0][..., j]
            diag.append(h_j)
        return torch.stack(diag, dim=-1)
