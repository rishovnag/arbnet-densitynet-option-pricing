"""arbnet: Architecturally No-Arbitrage Neural Pricers with Rough-Volatility Inductive Bias.

A research package implementing hard-constrained neural option pricing for index options,
with companion baselines (BS, Heston, rough Bergomi, Ackerer-style soft-penalty IV nets)
and a deep-hedging evaluation harness.
"""

__version__ = "0.1.0"
__all__ = ["models", "data", "losses", "arbitrage", "hedging", "eval", "calibration", "utils"]
