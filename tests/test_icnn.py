"""Test that ICNN is convex in the designated input direction."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from arbnet.models import ICNN


def test_icnn_convex():
    torch.manual_seed(0)
    net = ICNN(input_dim=1, hidden_dims=[16, 16], output_dim=1, context_dim=2)
    y = torch.linspace(-2, 2, 50).unsqueeze(-1)
    ctx = torch.randn(50, 2)
    diag = net.hessian_diag(y, ctx)
    assert (diag >= -1e-4).all(), f"convexity violated: min diag={diag.min().item()}"


def test_icnn_jensen():
    """Jensen's inequality: f((a+b)/2) <= (f(a) + f(b))/2 for a convex f."""
    torch.manual_seed(0)
    net = ICNN(input_dim=2, hidden_dims=[16, 16], output_dim=1)
    a = torch.randn(100, 2) * 2
    b = torch.randn(100, 2) * 2
    fa = net(a)
    fb = net(b)
    fmid = net(0.5 * (a + b))
    avg = 0.5 * (fa + fb)
    # Should hold for every row
    assert (fmid <= avg + 1e-4).all(), f"Jensen violated by {(fmid - avg).max().item()}"


if __name__ == "__main__":
    test_icnn_convex()
    test_icnn_jensen()
    print("test_icnn: OK")
