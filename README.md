# ArbNet

Reference implementation for **"Arbitrage-Free Deep Option Pricing via Convex Composition: Architectural Guarantees and Universal Approximation."**

## What this is

A neural architecture for European call surfaces that is **structurally free of static arbitrage** — not by penalty, but by construction. The architecture parameterises the call price as
```
C(K, T) = exp(-rT) * [ max(F_T - K, 0) + Delta(K, T) ]
F_T     = S * exp((r - q) * T)
Delta(K, T) = S * (1 - exp(-alpha * T))
            * sum_j  s_j * softplus(ICNN_j(K/S; ctx))
                         * softplus(MonotoneNet_j(T; ctx))
```
with an Input-Convex Network (ICNN, Amos et al., 2017) in strike and a Monotone network (Daniels & Velikova, 2010) in time-to-maturity. For every choice of weights and every context vector, the resulting surface satisfies four pointwise no-arbitrage conditions: (A1) convexity in K (no butterfly, via Breeden–Litzenberger); (A2) monotonicity in T of the carry-adjusted price (no calendar arbitrage); (A3) expiry boundary C(K, 0) = (S − K)⁺; (A4) lower bound by the discounted forward intrinsic. These guarantees are proved in Theorem 4.1 of the paper.

A second theorem (Theorem 5.1) shows the architecture is a **universal approximator** on its natural hypothesis class of K-convex, T-monotone, non-negative surfaces. Proposition 5.7 honestly states what the class excludes: surfaces with bell-shaped time value (including Black–Scholes) are not exactly representable, and this expressivity limit explains the empirical fit gap we observe.

## Repository layout

```
arbnet/
├── arbnet/                  # Main package
│   ├── models/
│   │   ├── icnn.py          # Input-Convex Neural Network (Amos et al. 2017)
│   │   ├── monotone.py      # Monotone NN with non-negative weights on monotone inputs
│   │   ├── composite.py     # ArbNet: intrinsic + convex-monotone correction
│   │   ├── baselines.py     # Black-Scholes, Heston (Lewis 2001 Fourier), Ackerer-Net
│   │   ├── rough_vol.py     # Rough Bergomi simulator (hybrid scheme)
│   │   └── svi_envelope.py  # SSVI envelope utility
│   ├── data/                # Synthetic surface generators (rough Bergomi, SSVI), filters,
│   │                        #   IV inversion (Newton with intrinsic guards), feature builder
│   ├── arbitrage/           # Static arbitrage detectors: butterfly (∂²C/∂K² ≥ 0),
│   │                        #   calendar (e^{rT} C non-decreasing in T), Roger Lee tail bound
│   ├── losses/              # Vega-weighted price RMSE, IV RMSE, soft no-arb penalty (for
│   │                        #   the Ackerer baseline only), CVaR hedging loss, composite
│   ├── hedging/             # One-step delta hedge under rough Bergomi paths, PnL stats
│   ├── eval/                # RMSE, bootstrap CI, Diebold-Mariano, hedge PnL stats
│   ├── calibration/         # Heston / rough Bergomi parameter calibration
│   ├── utils/               # Config, seed
│   └── train.py             # Training loop (price + IV + hedging losses)
├── scripts/
│   ├── smoke_test.py        # End-to-end pipeline check (~30s)
│   ├── train.py             # Train a single model
│   └── run_study.py         # Full experimental study (paper Table 1)
├── tests/                   # pytest unit tests (11 tests, ~5s)
├── results/                 # Output JSON from run_study.py
├── DATASETS.md              # Notes on data sources for the deferred Nifty backtest
├── requirements.txt
└── setup.py
```

## Quickstart

```bash
pip install -r requirements.txt        # PyTorch, NumPy, SciPy

# Unit tests (~5s)
python -m pytest tests/ -q

# End-to-end smoke test on a synthetic rough-Bergomi surface (~30s)
python scripts/smoke_test.py

# Reproduce the paper's main experiment (Table 1)
#   10 surfaces × 3 seeds × 150 epochs ≈ 5 minutes on CPU
python scripts/run_study.py --n_surfaces 10 --seeds_per_surface 3 --n_epochs 150
```

The study writes `results/study.json` with per-trial records and the aggregate
statistics reported in Table 1 of the paper.

## Reproducing the paper

`scripts/run_study.py` (i) draws 10 independent rough-Bergomi parameter tuples
(`H ∈ [0.07, 0.20]`, `η ∈ [1.0, 3.0]`, `ρ ∈ [-0.9, -0.3]`, `ξ₀ ∈ [0.02, 0.10]`),
(ii) simulates each surface with 4000 paths and prices 20 strikes × 5 maturities
({7, 14, 30, 60, 90} days) at Nifty-scale spot S = 20 000, r = 6.5%, q = 1.2%,
(iii) trains ArbNet, the Ackerer soft-penalty baseline, and Black–Scholes for
each (surface, seed) pair, and (iv) evaluates price RMSE, implied-volatility
RMSE, butterfly and calendar violation rates (tolerance 10⁻⁷ INR, one decade
above f32 noise), and one-step delta-hedged PnL tail statistics on 500 paths.

Headline numbers (n = 30 trials):

|                 | Price RMSE      | Butterfly | Calendar | Hedge std | CVaR₉₅  |
|-----------------|-----------------|-----------|----------|-----------|---------|
| **ArbNet**      | 134.27 ± 39.56  | 0.000%    | 0.000%   | 272.88    | 1128.32 |
| Ackerer (λ=1.0) | 77.85 ± 63.17   | 0.000%    | 0.813%   | 254.03    | 808.26  |
| Black–Scholes   | 37.44 ± 12.69   | 0.000%    | 1.037%*  | 161.72    | 435.07  |

*BS calendar violations are float-32 artifacts at deep wings (max magnitude
~10⁻⁴ INR per grid point), not economic arbitrages. See §6.3 of the paper.

The Ackerer baseline achieves a lower mean RMSE but never achieves zero
calendar violations across all (surface, seed) trials; ArbNet trades roughly
+57 INR of RMSE for exact, parameter-independent compliance. Paired t-test
across 10 surfaces: Δ = +56.7 INR, t = 2.31.

## Architecture details

The reference configuration:

- **Mixture components**: J = 4 experts.
- **ICNN per expert**: input dim 1 (scaled K/S), hidden widths [64, 64], output
  dim 1; non-negative weights on the z-path (`softplus`-parameterised), softplus
  activation throughout. Convexity in K is preserved by feeding `(K/S − 1) * 5`
  as input (an affine transform of K).
- **Monotone net per expert**: input is `(T, ctx)` with the T-column having
  non-negative weights; widths [32, 32].
- **Scalar gates**: `α = softplus(α̃)` controls the rate at which the correction
  grows from zero at T = 0; per-expert scales `s_j = softplus(s̃_j)`.
- **Outer softplus** on each expert ensures Δ ≥ 0; the `(1 − exp(−αT))`
  envelope enforces Δ(K, 0) = 0 exactly.

This is the construction proved arbitrage-free in Theorem 4.1.

## Out of scope (deferred to follow-up work)

- A walk-forward Nifty 50 backtest (≈ 1500 trading days, 2019–2024) with
  Diebold-Mariano-corrected significance tests and bootstrap CIs on hedge tail
  statistics. Data acquisition notes in `DATASETS.md`.
- Architectural Roger Lee wing-bound enforcement (currently the wing slope is
  bounded only softly by the ICNN's last-layer norm).
- Extension to bell-shaped time values via the difference-of-cones
  parametrisation sketched in §9.4 of the paper.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers ICNN convexity (numerical second-derivative check on random
inputs), monotone-net monotonicity, the static arbitrage detectors against
known-good and known-bad reference surfaces, and an end-to-end smoke test that
trains ArbNet for 40 epochs and verifies zero violations.

## License

Code released under the MIT License. The accompanying manuscript is © the
authors; please cite once published.

## References

Key methodological references (full bibliography in `paper/references.bib`):

- Amos, B., Xu, L., Kolter, J. Z. (2017). *Input convex neural networks*. ICML.
- Daniels, H., Velikova, M. (2010). *Monotone and partially monotone neural
  networks*. IEEE Transactions on Neural Networks 21(6).
- Ackerer, D., Tagasovska, N., Vatter, T. (2020). *Deep smoothing of the implied
  volatility surface*. NeurIPS.
- Bayer, C., Friz, P., Gatheral, J. (2016). *Pricing under rough volatility*.
  Quantitative Finance 16(6).
- Breeden, D. T., Litzenberger, R. H. (1978). *Prices of state-contingent claims
  implicit in option prices*. Journal of Business 51(4).
- Roper, M. (2010). *Arbitrage free implied volatility surfaces*. Preprint.
