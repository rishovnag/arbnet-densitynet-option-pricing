"""Test that MonotoneNet is non-decreasing in the designated input columns."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from arbnet.models import MonotoneNet


def test_monotone_increasing():
    torch.manual_seed(0)
    net = MonotoneNet(input_dim=3, hidden_dims=[16, 16], output_dim=1, monotone_cols=[0])
    base = torch.randn(50, 3)
    # Sweep column 0 upward
    for delta in [0.1, 0.5, 1.0, 2.0]:
        bumped = base.clone()
        bumped[:, 0] = bumped[:, 0] + delta
        with torch.no_grad():
            f_base = net(base)
            f_bump = net(bumped)
        assert (f_bump >= f_base - 1e-4).all(), \
            f"monotone violated for delta={delta}: max decrease={(f_base - f_bump).max().item()}"


def test_monotone_multiple_cols():
    torch.manual_seed(1)
    net = MonotoneNet(input_dim=4, hidden_dims=[8, 8], output_dim=1, monotone_cols=[0, 2])
    base = torch.randn(40, 4)
    # Bump column 2
    bumped = base.clone()
    bumped[:, 2] = bumped[:, 2] + 1.0
    with torch.no_grad():
        f_base = net(base)
        f_bump = net(bumped)
    assert (f_bump >= f_base - 1e-4).all()


if __name__ == "__main__":
    test_monotone_increasing()
    test_monotone_multiple_cols()
    print("test_monotone: OK")
