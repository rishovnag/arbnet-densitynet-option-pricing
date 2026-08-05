"""Test that the arbitrage detector flags known violations correctly."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from arbnet.arbitrage import (
    butterfly_violations,
    calendar_violations,
    roger_lee_tail_violations,
)


def test_butterfly_convex_pass():
    """A strictly convex C(K) should produce zero violations."""
    K = torch.linspace(80.0, 120.0, 21)
    # C(K) = max(S - K, 0) is piecewise linear, convex; use S=100 -> convex
    C = (100.0 - K).clamp(min=0.0) + 0.5 * (K - 100.0).pow(2) * 1e-3  # smooth convex
    v = butterfly_violations(C, K)
    assert (v <= 1e-4).all(), f"false butterfly violation: max={v.max().item()}"


def test_butterfly_concave_fail():
    """A concave C should produce many violations."""
    K = torch.linspace(80.0, 120.0, 21)
    C = -0.01 * (K - 100.0) ** 2 + 30.0  # concave parabola
    v = butterfly_violations(C, K)
    assert (v > 1e-4).any(), "concave price wasn't flagged"


def test_calendar_increasing_pass():
    """Strictly increasing w(T) should produce zero violations."""
    T = torch.linspace(0.01, 1.0, 20)
    w = 0.04 * T + 0.01 * T ** 2  # monotone increasing
    v = calendar_violations(w, T)
    assert (v <= 1e-6).all(), f"false calendar violation: max={v.max().item()}"


def test_calendar_decreasing_fail():
    """Decreasing w(T) should be flagged."""
    T = torch.linspace(0.01, 1.0, 20)
    w = 0.05 - 0.04 * T  # decreasing
    v = calendar_violations(w, T)
    assert (v > 0.0).any()


def test_lee_within_bound_pass():
    """Slope-1 total variance should pass (slope < 2)."""
    k = torch.linspace(-2.0, 2.0, 41)
    w = 0.04 + k.abs()  # slope = 1
    v = roger_lee_tail_violations(w, k, tail_fraction=0.3, bound=2.0)
    assert (v < 1e-3).all(), f"false Lee violation, v={v}"


def test_lee_above_bound_fail():
    """Slope-3 total variance should fail."""
    k = torch.linspace(-2.0, 2.0, 41)
    w = 0.04 + 3.0 * k.abs()  # slope = 3
    v = roger_lee_tail_violations(w, k, tail_fraction=0.3, bound=2.0)
    assert (v > 0.0).any()


if __name__ == "__main__":
    test_butterfly_convex_pass()
    test_butterfly_concave_fail()
    test_calendar_increasing_pass()
    test_calendar_decreasing_fail()
    test_lee_within_bound_pass()
    test_lee_above_bound_fail()
    print("test_arbitrage: OK")
