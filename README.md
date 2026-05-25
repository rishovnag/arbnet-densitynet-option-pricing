# ArbNet

**Arbitrage-Free Deep Option Pricing via Convex Composition** — a neural
architecture for European call surfaces that is free of butterfly and calendar
static arbitrage *by construction*, not by penalty. This repository is the
reference implementation for the accompanying manuscript ("Arbitrage-Free Deep
Option Pricing via Convex Composition: Architectural Guarantees and Universal
Approximation"), and it ships with the real NSE Nifty 50 options data needed to
reproduce every number below.

---

## Table of contents

- [What ArbNet is](#what-arbnet-is)
- [The guarantee — and its precise scope](#the-guarantee--and-its-precise-scope)
- [Headline results](#headline-results)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [The bundled data](#the-bundled-data)
- [The real-data pipeline](#the-real-data-pipeline)
- [Architecture details](#architecture-details)
- [Models and baselines](#models-and-baselines)
- [Reproducing the studies](#reproducing-the-studies)
- [Tests](#tests)
- [Honest limitations](#honest-limitations)
- [License](#license)
- [References](#references)

---

## What ArbNet is

ArbNet parameterises the European call price surface as a discounted intrinsic
payoff plus a non-negative, convex-in-strike, monotone-in-maturity correction:

```
C(K, T)      = exp(-rT) * [ max(F_T - K, 0) + Delta(K, T) ]
F_T          = S * exp((r - q) * T)
Delta(K, T)  = S * (1 - exp(-alpha * T))
             * sum_j  s_j * softplus( ICNN_j(K/S ; ctx) )
                          * softplus( MonotoneNet_j(T ; ctx) )
```

- **`ICNN_j`** — an Input-Convex Neural Network (Amos et al., 2017) that is
  convex in the (affine-transformed) strike. `softplus` of a convex function is
  convex, so each expert is convex in `K`.
- **`MonotoneNet_j`** — a partially-monotone network (Daniels & Velikova, 2010)
  with non-negative weights on the maturity column, hence non-decreasing in `T`.
- **`(1 - exp(-alpha T))`** — an envelope that forces `Delta(K, 0) = 0`, so the
  surface collapses exactly onto the intrinsic payoff at expiry.
- **outer `softplus`** — keeps every expert, and therefore `Delta`, non-negative.

Because convexity, monotonicity and non-negativity are *structural* properties
of the parameterisation, they hold for **every** choice of network weights and
**every** context vector — before training, during training, and after. There
is no soft penalty and no tolerance to tune.

A companion universal-approximation result (Theorem 5.1 of the paper) shows the
architecture is dense in its natural hypothesis class — K-convex, T-monotone,
non-negative surfaces. Proposition 5.7 states honestly what that class
*excludes*: surfaces with bell-shaped time value (Black–Scholes among them) are
not exactly representable, which is the source of the fit gap reported below.

## The guarantee — and its precise scope

For any weights, ArbNet satisfies four pointwise no-arbitrage conditions
(Theorem 4.1):

| | Condition | Enforced by |
|---|---|---|
| **A1** | Convexity in `K` — no butterfly arbitrage (Breeden–Litzenberger) | ICNN convexity |
| **A2** | Carry-adjusted price non-decreasing in `T` — no calendar arbitrage | MonotoneNet + envelope |
| **A3** | Expiry boundary `C(K, 0) = (S − K)⁺` | `(1 - exp(-alpha T))` envelope |
| **A4** | Lower bound by the discounted forward intrinsic | `Delta ≥ 0` |

**What the guarantee does *not* cover.** The Roger Lee wing/tail bound — a
*fifth* static no-arbitrage condition on the asymptotic slope of total implied
variance — is **not** enforced architecturally. ArbNet's wing slope is bounded
only softly, by the ICNN's last-layer norm. The codebase therefore reports
butterfly and calendar compliance as **`guaranteed`** and the Roger Lee tail as
a separate **`monitored`** diagnostic. Read every "arbitrage-free by
construction" claim as scoped to A1–A4.

## Headline results

Two studies, both reproducible from this repository.

### Synthetic study — `results/study.json`

10 rough-Bergomi surfaces × 3 seeds × 150 epochs (`scripts/run_study.py`):

| Model | Price RMSE | Butterfly | Calendar | Hedge std | CVaR₉₅ |
|---|---|---|---|---|---|
| **ArbNet** | 134.3 ± 39.6 | **0.000%** | **0.000%** | 272.9 | 1128 |
| Ackerer (λ=1) | 78.1 ± 62.5 | 0.154% (max 4.6%) | 0.000% | 255.5 | 834 |
| Black–Scholes | 37.4 ± 12.7 | 0.000% | 0.000% | 161.8 | 434 |

Paired ArbNet−Ackerer RMSE test: Δ = +56.2 INR, t = 3.71 (n = 30).

### Real-data study — `results/nse_study_full.json`

Walk-forward over **all 1,359 NSE trading days, 2019-01-01 → 2024-07-05**
(`scripts/train_nse.py`), one independent fit per day, 10 macro/auxiliary
context features:

| Model | Price RMSE (95% CI) | Butterfly + Calendar | Tail (monitored) | Hedge std / CVaR₉₅ |
|---|---|---|---|---|
| **ArbNet** | 95.2 (93.7–96.6) | **0 violations / 1359 days** | 0.14% mean | 175.9 / 728 |
| Ackerer (λ=1) | 36.2 (35.1–37.3) | **violations on 961 / 1359 days** | 0.52% mean | 125.1 / 338 |
| Black–Scholes | 39.7 (38.8–40.6) | 0 / 1359 | 0.00% | 124.2 / 323 |

Diebold–Mariano (HAC-corrected, 7-lag Newey–West): ArbNet vs Ackerer
dm = 32.0, ArbNet vs BS dm = 31.2 (both p ≈ 0).

**Reading the results.** ArbNet holds the architectural guarantee *exactly* on
every real trading day, including the March-2020 COVID crash; the soft-penalty
Ackerer baseline violates butterfly/calendar on 71% of real days (up to 31.6%).
ArbNet pays for that exactness with a higher, regime-dependent fit error
(by year: 63 in 2019, ~115 in the high-vol 2021–22, 102 in 2024) — the
expressivity gap that Proposition 5.7 predicts. The Roger Lee tail is breached
only in the crash window, and only because it is *monitored*, not enforced.

## Repository layout

```
arbnet/
├── README.md                 # this file
├── DATASETS.md                # data sources, schemas, provenance
├── LICENSE                    # MIT (code); data terms noted within
├── requirements.txt
├── setup.py
├── .gitignore
│
├── arbnet/                    # the importable package
│   ├── models/
│   │   ├── icnn.py            # Input-Convex Neural Network
│   │   ├── monotone.py        # partially-monotone network
│   │   ├── composite.py       # ArbNet: intrinsic + convex-monotone correction
│   │   ├── baselines.py       # Black–Scholes, Heston (Lewis 2001), Ackerer net
│   │   ├── rough_vol.py       # rough Bergomi Monte-Carlo simulator
│   │   └── svi_envelope.py    # SSVI total-variance utility
│   ├── data/
│   │   ├── loaders.py         # NSE bhavcopy parser  (OptionsSnapshot)
│   │   ├── real_data.py       # bundled-data loaders (spot, rates, context …)
│   │   ├── synthetic.py       # SSVI + rough-Bergomi surface generators
│   │   ├── filters.py         # quality filters (stale / illiquid / no-arb)
│   │   ├── features.py        # tensor feature builder
│   │   └── iv.py              # vectorised Newton implied-vol inverter
│   ├── losses/                # price RMSE, IV RMSE, soft no-arb penalty, CVaR
│   ├── arbitrage/             # butterfly / calendar / Roger Lee detectors
│   ├── hedging/               # delta hedge + deep-hedge PnL
│   ├── eval/                  # RMSE, bootstrap CI, Diebold–Mariano, tables
│   ├── calibration/           # Heston / rough-Bergomi parameter fitting
│   ├── utils/                 # run config, seeding
│   └── train.py               # the training loop
│
├── scripts/
│   ├── smoke_test.py          # end-to-end pipeline check on synthetic data (~30 s)
│   ├── train.py               # train a single model on one surface
│   ├── train_nse.py           # walk-forward study on the real NSE data
│   ├── run_study.py           # synthetic rough-Bergomi study (paper Table 1)
│   ├── check_data.py          # validate the bundled data tree against the loaders
│   └── prepare_data.py        # (optional) re-fetch raw data from NSE / RBI / yfinance
│
├── data/                      # bundled real market data — see DATASETS.md
│   ├── nse/
│   │   ├── fo_bhavcopy/        # 1359 daily F&O bhavcopies (NIFTY + BANKNIFTY)
│   │   └── nifty50_spot.csv    # official Nifty 50 daily close
│   ├── rates/                 # 91-day T-bill auctions, NIFTY 50 dividend yield
│   ├── auxiliary/             # India VIX, USD/INR, FII/DII net flows
│   └── macro/                 # CPI / WPI / IIP series + macro event calendar
│
├── tests/                     # pytest unit tests (11 tests, ~5 s)
└── results/                   # study.json (synthetic), nse_study_full.json (real)
```

## Installation

Python ≥ 3.9.

```bash
git clone <repo-url> && cd arbnet
pip install -r requirements.txt          # torch, numpy, pandas, scipy, openpyxl
# or, as an editable package:
pip install -e ".[data]"
```

`openpyxl` is needed to read the 91-day T-bill auctions spreadsheet; without it
the risk-free rate silently falls back to a 6.5% constant. The optional
`prepare_data.py` re-fetch script additionally needs `requests`, `yfinance` and
`tqdm` (`pip install -e ".[fetch]"`).

## Quickstart

```bash
# 1. Unit tests (~5 s)
python -m pytest tests/ -q

# 2. Validate the bundled data tree against the loaders
python scripts/check_data.py

# 3. End-to-end smoke test on a synthetic rough-Bergomi surface (~30 s)
python scripts/smoke_test.py

# 4. Synthetic study — reproduces Table 1 (~2 min on CPU)
python scripts/run_study.py --n_surfaces 10 --seeds_per_surface 3 --n_epochs 150 \
    --out results/study.json

# 5. Real-data study — full 1359-day walk-forward (hours on CPU; use --stride to subsample)
python scripts/train_nse.py --models arbnet ackerer bs --n_epochs 150 \
    --out results/nse_study_full.json
```

Using the package directly:

```python
from arbnet.data import load_snapshot, apply_quality_filters, FilterConfig, build_features
from arbnet.models import ArbNet
from arbnet.utils import default_arbnet_config, RunConfig
from arbnet.train import train_pricer

snap = load_snapshot("2024-07-05").call_subset()        # real NIFTY calls for that day
filtered, summary = apply_quality_filters(snap, FilterConfig())
feats = build_features(filtered)

model = ArbNet(default_arbnet_config())
train_pricer(model, feats, RunConfig(n_epochs=150, lr=1e-2))
```

## The bundled data

`data/` ships the real Indian-market data so the studies run with no downloads.
Full schemas, sources and provenance are in **[DATASETS.md](DATASETS.md)**;
in brief:

| Path | Contents | Used for |
|---|---|---|
| `nse/fo_bhavcopy/fo_*.csv` | 1359 daily NSE F&O bhavcopies, 2019–2024 | NIFTY option quotes (training); BANKNIFTY (context) |
| `nse/nifty50_spot.csv` | official Nifty 50 daily close | spot `S` |
| `rates/Auctions of 91-Day…T-bills.xlsx` | RBI 91-day T-bill cut-off yields | risk-free rate `r` |
| `rates/NIFTY 50-yield-*.csv` | NIFTY 50 dividend yield series | dividend yield `q` |
| `auxiliary/india_vix.csv` | India VIX daily close | context feature |
| `auxiliary/usdinr.csv` | USD/INR daily close | context feature |
| `auxiliary/fii_dii.csv` | daily FII / DII net cash flows | context features |
| `macro/{cpi,wpi,iip}.csv` | CPI / WPI / IIP monthly series | context features |
| `macro/india_macro_calendar_extended.csv` | RBI MPC, Budget, CPI/WPI/IIP release dates | context feature |

**Note on the bhavcopies.** The raw NSE bhavcopy lists every F&O instrument
(~34k rows/day). The committed files are pre-filtered to the only symbols this
project uses — **NIFTY and BANKNIFTY** — which shrinks the dataset roughly 10×
(~4.9 GB → ~430 MB) while leaving the loaders' behaviour unchanged. To work with
the full multi-asset bhavcopy, re-fetch via `scripts/prepare_data.py`. At ~430 MB
the `data/` directory is still sizeable; consider Git LFS when hosting.

## The real-data pipeline

`scripts/train_nse.py` runs, for each trading day independently:

```
bhavcopy ─▶ load_snapshot ─▶ call_subset ─▶ apply_quality_filters
            (spot: official close → nearest future → put-call parity;
             r from T-bills; q from the dividend-yield series;
             corrupted days are skipped, never fudged)
         ─▶ implied_vol_newton ─▶ build_features (+ 10-feature context)
         ─▶ train_pricer  (ArbNet / Ackerer / Black–Scholes)
         ─▶ evaluate: price RMSE, IV RMSE, static-arbitrage report
         ─▶ delta-hedge PnL (std, CVaR₉₅; guarded against degenerate vol)
then ─▶ per-model aggregates with bootstrap 95% CIs
     ─▶ HAC-corrected Diebold–Mariano tests
     ─▶ a data-quality report (every skipped day, with the reason)
```

The **context vector** (fed to ArbNet's ICNN and monotone modules; ArbNet stays
arbitrage-free for any context) has 10 features, all look-ahead-safe — macro
prints are lagged to their real publication dates:

`india_vix`, `banknifty_iv`, `realized_vol_1w`, `usdinr`, `fii_net_cr`,
`dii_net_cr`, `cpi_inflation`, `wpi_inflation`, `iip_growth`,
`days_to_macro_event`.

Useful flags: `--start / --end / --stride` to subsample the date range,
`--max_days` to cap, `--no_context` for an ablation that matches the synthetic
study, `--models` to choose any subset of `arbnet ackerer bs`.

## Architecture details

Reference configuration (`arbnet.utils.default_arbnet_config`):

- **Mixture** — `J = 6` experts.
- **ICNN per expert** — input dim 1 (the scaled strike `(K/S − 1)·5`, an affine
  transform of `K`, so convexity in `K` is preserved), hidden widths `[64, 64]`,
  softplus activations, non-negative `z`-path weights (`softplus`-parameterised).
- **MonotoneNet per expert** — input `(T, context)`, hidden widths `[32, 32]`,
  non-negative weights on the `T` column.
- **Scalar gates** — `alpha = softplus(α̃)` sets how fast the correction grows
  from zero at `T = 0`; per-expert scales `s_j = softplus(s̃_j)`.

Training (`arbnet.train.train_pricer`) is Adam on a vega-weightable price-RMSE
objective, optionally with IV-RMSE and a soft no-arbitrage penalty — the penalty
is used **only** by the Ackerer baseline; ArbNet never needs it.

## Models and baselines

- **`ArbNet`** (`models/composite.py`) — the architecturally-constrained pricer.
- **`AckererSoftPenaltyNet`** (`models/baselines.py`) — an unconstrained MLP for
  total variance trained with a *soft* no-arbitrage penalty (Ackerer et al.,
  2020); the primary competitor.
- **`BlackScholesPricer`** — one ATM-fitted volatility; the unconstrained fit
  floor.
- **`HestonPricer`** — Heston (1993) via the Lewis (2001) Fourier integral;
  with `arbnet.calibration.calibrate_heston`.
- **`RoughBergomiSimulator`** (`models/rough_vol.py`) — the rough-volatility
  Monte-Carlo engine used both to generate synthetic ground-truth surfaces and
  to simulate hedging paths.

## Reproducing the studies

**Synthetic (Table 1).** `run_study.py` draws 10 rough-Bergomi parameter tuples
(`H ∈ [0.07,0.20]`, `η ∈ [1.0,3.0]`, `ρ ∈ [−0.9,−0.3]`, `ξ₀ ∈ [0.02,0.10]`),
simulates each with 4000 paths over 20 strikes × 5 maturities at Nifty-scale
`S = 20000`, trains all three models for each (surface, seed) pair, and writes
per-trial records plus aggregates to `results/study.json`.

**Real (1359-day walk-forward).** `train_nse.py` as described above, writing
`results/nse_study_full.json`.

Both arbitrage grids are evaluated in **float64** — at Nifty-scale prices the
float32 differencing noise (~10⁻⁴ INR) sits three decades above the 10⁻⁷
violation tolerance and would manufacture spurious violations.

## Tests

```bash
python -m pytest tests/ -q          # 11 tests, ~5 s
```

The suite covers ICNN convexity (a numerical second-derivative check on random
inputs), MonotoneNet monotonicity, the static-arbitrage detectors against
known-good and known-bad reference surfaces, and an end-to-end smoke test that
trains ArbNet and verifies zero violations.

## Honest limitations

- **Expressivity gap.** ArbNet's hypothesis class cannot represent bell-shaped
  time value, so it carries a real, regime-dependent RMSE penalty versus
  unconstrained baselines. This is intended: the contribution is *exact*
  arbitrage-freeness, not best fit.
- **Roger Lee tail not enforced.** Only butterfly and calendar are architectural
  (see [scope](#the-guarantee--and-its-precise-scope)).
- **Significance.** The real study's per-day errors are serially correlated;
  use the HAC-corrected Diebold–Mariano statistics, not the (retained but
  labelled) uncorrected paired t-test.
- **Bell-shaped time value.** A difference-of-cones extension is sketched in
  §9.4 of the paper and left to follow-up work.

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The bundled
market data remains the property of its sources (NSE, RBI, MoSPI, and others)
and is redistributed only for academic reproducibility. The accompanying
manuscript is © the authors; please cite once published.

## References

- Amos, B., Xu, L., Kolter, J. Z. (2017). *Input convex neural networks*. ICML.
- Daniels, H., Velikova, M. (2010). *Monotone and partially monotone neural
  networks*. IEEE Transactions on Neural Networks 21(6).
- Ackerer, D., Tagasovska, N., Vatter, T. (2020). *Deep smoothing of the implied
  volatility surface*. NeurIPS.
- Bayer, C., Friz, P., Gatheral, J. (2016). *Pricing under rough volatility*.
  Quantitative Finance 16(6).
- Breeden, D. T., Litzenberger, R. H. (1978). *Prices of state-contingent claims
  implicit in option prices*. Journal of Business 51(4).
- Lee, R. W. (2004). *The moment formula for implied volatility at extreme
  strikes*. Mathematical Finance 14(3).
- Diebold, F. X., Mariano, R. S. (1995). *Comparing predictive accuracy*.
  Journal of Business & Economic Statistics 13(3).
- Roper, M. (2010). *Arbitrage-free implied volatility surfaces*. Preprint.
