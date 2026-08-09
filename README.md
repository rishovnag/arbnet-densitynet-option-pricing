# ArbNet / DensityNet

**Arbitrage-free deep option pricing by construction: from convex composition
to martingale-mixture densities** — two neural architectures for European call
surfaces whose no-arbitrage properties hold *by construction*, not by penalty,
with certificates of deliberately different strength. This repository is the
reference implementation for the accompanying manuscript, and it ships with the
real NSE Nifty 50 options data needed to reproduce every number below.

- **ArbNet** (`arbnet/models/composite.py`) certifies the four canonical
  pointwise conditions — butterfly, calendar, expiry, forward-intrinsic bound
  — for **every** weight vector, via input-convex networks in strike and
  monotone networks in maturity. The paper proves this certificate is also the
  construction's **ceiling**: the four conditions do not exhaust static
  no-arbitrage, and every fully arbitrage-free surface with nonzero time value
  lies *outside* the hypothesis class. ArbNet is the instrument that locates
  where elementary structural compliance binds.
- **DensityNet** (`arbnet/models/density.py`) — the main contribution — prices
  against a learned martingale mixture of lognormals in forward-moneyness
  (fixed component means, one shared increasing variance clock) and satisfies
  the **complete** static no-arbitrage characterisation for every parameter
  vector, with a smooth atom-free density and a closed-form single-pass price.

---

## Table of contents

- [Headline results (held-out study)](#headline-results-held-out-study)
- [The two guarantees — and their precise scope](#the-two-guarantees--and-their-precise-scope)
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

## Headline results (held-out study)

Two studies, both reproducible from this repository with pinned seeds.

### Real-data study — `results/nse_study_v3_holdout.json`

All **1,359 NSE trading days, 2019-01-01 → 2024-07-05**, one independent fit
per day. Each day's filtered cross-section is ranked by `(T, K)` and split by
parity: models fit on the odd half, all primary metrics are evaluated on the
**held-out even half** (`--holdout odd_even`). Diebold–Mariano statistics are
HAC-corrected paired comparisons of daily held-out losses.

| Model | Held-out price RMSE (95% CI) | Held-out IV RMSE | (A1)–(A4) violation days | Complete-list compliant days |
|---|---|---|---|---|
| **DensityNet** | **38.99** (36.4–41.8) | 0.075 | **0 / 1359** | **1359 / 1359** |
| ArbNet | 96.77 (92.9–100.7) | 0.096 | **0 / 1359** | 0 / 1359 |
| Ackerer (λ=1) | 43.62 (40.9–46.3) | 0.065 | 726 / 1359 | 57 / 1359 |
| Black–Scholes | 41.08 (38.6–43.6) | **0.063** | 0 / 1359 | 1359 / 1359 |

DensityNet is **significantly the best held-out price fit of every model
tested** (DM −12.1 vs Ackerer, −9.5 vs BS, −33.0 vs ArbNet, all p < 1e-3)
while satisfying the complete static no-arbitrage characterisation on every
day, verified adversarially on forward-refined off-training float64 grids.
The reversal of the in-sample tie (34.2 vs 35.4 INR) is a parsimony effect:
from fitted to held-out half the 18-coefficient DensityNet degrades by 4.8 INR
while the 8.6k-parameter baseline degrades by 8.3.

The **extended compliance diagnostic** (`arbnet/arbitrage/full_compliance.py`;
strike monotonicity, the call-spread/digital bound, the covered-call upper
envelope, and the zero-strike boundary, on `K/S ∈ [1e-3, 4]`) confirms the
paper's incompleteness proposition with no exceptions: ArbNet violates the
beyond-certificate conditions **inside the traded band on all 1,359 days** —
924 days with the call increasing in strike just above the forward (median
first violation at `K/S = 1.003`), 435 days with the digital bound broken and
a positive zero-strike boundary gap — while DensityNet and Black–Scholes pass
the complete list at every tested point. In volatility space the ordering
flips: out of sample the one-parameter Black–Scholes model is the IV-space
frontier, and DensityNet's wing gap (0.240 vega-weighted vs 0.053) is the
open problem the paper states.

### Synthetic study — `results/study_v2_full.json`

10 rough-Bergomi surfaces × 3 seeds × 150 epochs (`scripts/run_study.py`;
in-sample by design — the synthetic study is not strike-split):

| Model | Price RMSE | Butterfly | Calendar |
|---|---|---|---|
| **DensityNet** | **27.0 ± 9.1** | **0.000%** | **0.000%** |
| ArbNet | 134.3 ± 39.6 | **0.000%** | **0.000%** |
| Ackerer (λ=1) | 79.0 ± 62.7 | 0.051% | 0.000% |
| Black–Scholes | 37.5 ± 12.7 | 0.000% | 0.000% |

A λ-sweep over four decades (`results/lambda_sweep_*.json`) shows the
soft-penalty baseline's calendar violations can be driven to zero but its
butterfly rate floors near 3% of grid points at **any** penalty weight —
the guarantee is not purchasable by penalisation.

## The two guarantees — and their precise scope

For any weights, **ArbNet** satisfies four pointwise no-arbitrage conditions:

| | Condition | Enforced by |
|---|---|---|
| **A1** | Convexity in `K` — no butterfly arbitrage (Breeden–Litzenberger) | ICNN convexity |
| **A2** | Carry-adjusted price non-decreasing in `T` — no calendar arbitrage | MonotoneNet + envelope |
| **A3** | Expiry boundary `C(K, 0) = (S − K)⁺` | `(1 - exp(-alpha T))` envelope |
| **A4** | Lower bound by the discounted forward intrinsic | `Delta ≥ 0` |

**What (A1)–(A4) does *not* cover — and provably cannot.** The complete
characterisation of a static-arbitrage-free surface additionally requires
strike monotonicity (`∂C/∂K ≤ 0`), the call-spread/digital bound
(`−∂C/∂K ≤ e^{−rT}`), and the covered-call upper envelope with the exact
zero-strike boundary `C(0,T) = S e^{−qT}`. A convex forward time value that
respects those boundary conditions must vanish identically (a three-line
argument in the paper), so **every non-trivial ArbNet surface violates them**
— the implied measure carries a Dirac atom at the forward *and* excess total
mass. The codebase therefore reports (A1)–(A4) as **`guaranteed`**, the
extended list via **`full_*`** diagnostics, and the Roger Lee tail bound as
**`monitored`**. Read every ArbNet "arbitrage-free by construction" claim as
scoped to A1–A4.

**DensityNet** is the resolution: it prices as the discounted expectation of
`(S_T − K)⁺` under a genuine probability law with mean `F_T`, so *every*
condition of the complete characterisation holds for every parameter vector —
butterfly with a smooth atom-free density, calendar at fixed forward-moneyness
(the exact model-free condition, via the convex order of the martingale
factorisation `x_T = m(θ)·M_T`), the digital bound, the upper envelope, and
the zero-strike boundary exactly. Its one structural concession is an
approximate expiry collapse (a thin residual smile as `T → 0` instead of the
exact kink), which is a boundary approximation, not an arbitrage.

## Repository layout

```
arbnet/
├── README.md                  # this file
├── DATASETS.md                # data sources, schemas, provenance
├── LICENSE                    # MIT (code); data terms noted within
├── requirements.txt
├── setup.py
│
├── arbnet/                    # the importable package
│   ├── models/
│   │   ├── icnn.py            # Input-Convex Neural Network
│   │   ├── monotone.py        # partially-monotone network
│   │   ├── composite.py       # ArbNet: intrinsic + convex-monotone correction
│   │   ├── density.py         # DensityNet: martingale lognormal mixture
│   │   ├── baselines.py       # Black–Scholes, Heston (Lewis 2001), Ackerer net
│   │   ├── rough_vol.py       # rough Bergomi Monte-Carlo simulator
│   │   └── svi_envelope.py    # SSVI total-variance utility
│   ├── data/                  # NSE loaders, filters, features, Newton IV
│   ├── losses/                # price RMSE, IV RMSE, soft no-arb penalty, CVaR
│   ├── arbitrage/
│   │   ├── checks.py          # butterfly / calendar / Roger Lee (A1–A4 grid report)
│   │   ├── adversarial.py     # forward-refined float64 off-training-grid recheck
│   │   └── full_compliance.py # extended list: monotonicity, digital bound,
│   │                          #   upper envelope, K→0 boundary (K/S ∈ [1e-3, 4])
│   ├── hedging/               # delta hedge + deep-hedge PnL
│   ├── eval/                  # RMSE, block-bootstrap CI, Diebold–Mariano
│   ├── calibration/           # Heston / rough-Bergomi parameter fitting
│   ├── utils/                 # run config, seeding
│   └── train.py               # the training loop
│
├── scripts/
│   ├── smoke_test.py          # end-to-end pipeline check (~30 s)
│   ├── train_nse.py           # real-data study: per-day fits, --holdout odd_even,
│   │                          #   --save_params, extended compliance per record
│   ├── run_study.py           # synthetic rough-Bergomi study
│   ├── lambda_sweep.py        # penalty-weight sweep for the Ackerer baseline
│   ├── probe_full_compliance.py  # 7-day regime-spanning extended-list probe
│   ├── check_data.py          # validate the bundled data tree
│   └── prepare_data.py        # (optional) re-fetch raw data
│
├── data/                      # bundled real market data — see DATASETS.md
├── tests/                     # pytest unit tests
└── results/                   # committed study artifacts:
    ├── nse_study_v3_holdout.json            # PRIMARY: held-out real study (+ bump)
    ├── nse_study_v3_holdout_nocontext.json  # context ablation
    ├── study_v2_full.json                   # synthetic study
    └── lambda_sweep_{plain,matched}_corrected.json
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
# 1. Unit tests
python -m pytest tests/ -q

# 2. Validate the bundled data tree against the loaders
python scripts/check_data.py

# 3. End-to-end smoke test on a synthetic rough-Bergomi surface (~30 s)
python scripts/smoke_test.py

# 4. Synthetic study (~2 min on CPU)
python scripts/run_study.py --n_surfaces 10 --seeds_per_surface 3 --n_epochs 150 \
    --out results/study_v2_full.json

# 5. Real-data study — the paper's primary artifact (roughly 12 CPU-hours;
#    use --stride to subsample; per-model resumable with --resume)
python scripts/train_nse.py \
    --models arbnet ackerer bs ackerer_matched arbnet_density arbnet_bump \
    --n_epochs 150 --holdout odd_even --save_params results/params_v3 \
    --out results/nse_study_v3_holdout.json
```

Using the package directly:

```python
from arbnet.data import load_snapshot, apply_quality_filters, FilterConfig, build_features
from arbnet.models import DensityNet, DensityNetConfig
from arbnet.utils import RunConfig
from arbnet.train import train_pricer
from arbnet.arbitrage.full_compliance import full_compliance_report

snap = load_snapshot("2024-07-05").call_subset()        # real NIFTY calls for that day
filtered, summary = apply_quality_filters(snap, FilterConfig())
feats = build_features(filtered)

model = DensityNet(DensityNetConfig(n_components=8))
train_pricer(model, feats, RunConfig(n_epochs=150, lr=1e-2))

rep = full_compliance_report(model, S=float(snap.spot),
                             r=float(snap.risk_free_rate),
                             q=float(snap.dividend_yield))
print(rep)          # complete-list compliance, bucketed by moneyness
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
         ─▶ odd/even (T,K)-rank split          [--holdout odd_even]
         ─▶ train_pricer on the odd half
         ─▶ evaluate on the held-out even half: price / IV / vega RMSE
            (in-sample values kept alongside as *_insample)
         ─▶ static-arbitrage report (A1–A4, float64 grid)
         ─▶ adversarial forward-refined off-training recheck
         ─▶ extended full-list compliance report (full_* fields)
         ─▶ per-(day, model) parameter checkpoint   [--save_params DIR]
         ─▶ delta-hedge PnL (std, CVaR₉₅; guarded against degenerate vol)
then ─▶ per-model aggregates with moving-block-bootstrap 95% CIs
     ─▶ HAC-corrected Diebold–Mariano tests on the held-out losses
     ─▶ a data-quality report (every skipped day, with the reason)
```

The **context vector** (fed to the context-aware models; both architectures
stay compliant for any context) has 10 look-ahead-safe features — macro prints
are joined as-of their real publication dates:

`india_vix`, `banknifty_iv`, `realized_vol_1w`, `usdinr`, `fii_net_cr`,
`dii_net_cr`, `cpi_inflation`, `wpi_inflation`, `iip_growth`,
`days_to_macro_event`.

Useful flags: `--start / --end / --stride / --max_days` to subsample,
`--no_context` for the ablation, `--holdout odd_even` for the held-out split,
`--save_params DIR` for checkpoints, `--resume` to add models to an existing
output file without retraining finished ones, `--models` to choose any subset
of `arbnet ackerer bs ackerer_matched arbnet_density arbnet_bump`.

## Architecture details

**ArbNet** (`arbnet.utils.default_arbnet_config`): `J = 6` experts; per expert
an ICNN on the scaled strike `(K/S − 1)·5` with hidden widths `[64, 64]` and a
MonotoneNet on `(T, context)` with widths `[32, 32]`; softplus activations;
non-negative pass-through weights via softplus reparameterisation; scalar
gates `alpha = softplus(α̃)` (expiry envelope rate) and `s_j = softplus(s̃_j)`.
~33k parameters context-free, ~43k with the 10-feature context.

**DensityNet** (`DensityNetConfig(n_components=8)`): a small head maps the
context (or a raw parameter block) to `M = 8` softmax weights, `M` fixed
component means normalised to mixture mean 1 (the martingale property), and
two clock coefficients giving `σ²(T) = softplus(c₀)·T + softplus(c₁)·T²` —
18 effective per-surface coefficients. The price is a closed-form mixture of
Black calls; Greeks by autodifferentiation.

Training (`arbnet.train.train_pricer`) is Adam on a price-RMSE objective; the
soft no-arbitrage penalty is used **only** by the Ackerer baselines — neither
ArbNet nor DensityNet ever needs it.

## Models and baselines

- **`DensityNet`** (`models/density.py`) — the fully arbitrage-free
  martingale-mixture pricer; the paper's main contribution.
- **`ArbNet`** (`models/composite.py`) — the (A1)–(A4)-certified
  convex-composition pricer; the paper's diagnostic instrument. With
  `use_concave_bump=True` it becomes the negative control that trades
  butterfly compliance for at-the-money fit.
- **`AckererSoftPenaltyNet`** (`models/baselines.py`) — unconstrained MLP with
  a *soft* no-arbitrage penalty (Ackerer et al., 2020); the primary
  competitor, plus a capacity- and context-matched variant.
- **`BlackScholesPricer`** — one ATM-fitted volatility; out of sample, the
  volatility-space frontier of the study.
- **`HestonPricer`** — Heston (1993) via the Lewis (2001) Fourier integral.
- **`RoughBergomiSimulator`** (`models/rough_vol.py`) — synthetic surface
  generation and hedging paths.

## Reproducing the studies

Every random seed is pinned (simulator, data-loader shuffle, PyTorch/NumPy
generators). The committed artifacts in `results/` were produced by exactly
the Quickstart commands above; the context ablation adds `--no_context` and
writes `results/nse_study_v3_holdout_nocontext.json`; the λ-sweep is
`scripts/lambda_sweep.py --stride 20 [--plain]`. Both arbitrage grids and all
compliance diagnostics run in **float64** — at Nifty-scale prices, float32
differencing noise sits three decades above the violation tolerance.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers ICNN convexity, MonotoneNet monotonicity, the
static-arbitrage detectors against known-good and known-bad surfaces, the
DensityNet theorem (martingale normalisation, adversarial-clean across random
parameters, convex-order calendar, expiry boundary), a Heston regression
guard, and an end-to-end smoke test.

## Honest limitations

- **ArbNet's certificate is (A1)–(A4) and provably cannot be more.** Every
  non-trivial ArbNet surface violates strike monotonicity or the digital and
  covered-call bounds — in the real study, inside the traded band on all
  1,359 days. That incompleteness is a theorem in the paper, and the reason
  DensityNet exists.
- **Volatility-space gap.** DensityNet is the best held-out *price* fit but
  significantly worse than the unconstrained baseline (and BS) in IV space;
  the wing gap traces to the shared variance clock, and closing it inside the
  guarantee is the paper's stated open problem.
- **Roger Lee tail not enforced** by either architecture (monitored only).
- **No forecasting claim.** The held-out split is across strikes within a
  day; each day remains an independent calibration. Use the HAC-corrected
  Diebold–Mariano statistics, not the uncorrected paired t-test.
- **Single market.** All real-data magnitudes are NSE Nifty 50, 2019–2024.

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The bundled
market data remains the property of its sources (NSE, RBI, MoSPI, and others)
and is redistributed only for academic reproducibility. The accompanying
manuscript is © the authors; please cite once published.

## References

- Amos, B., Xu, L., Kolter, J. Z. (2017). *Input convex neural networks*. ICML.
- Chen, Y., Shi, Y., Zhang, B. (2019). *Optimal control via neural networks:
  a convex approach*. ICLR.
- Daniels, H., Velikova, M. (2010). *Monotone and partially monotone neural
  networks*. IEEE Transactions on Neural Networks 21(6).
- Ackerer, D., Tagasovska, N., Vatter, T. (2020). *Deep smoothing of the implied
  volatility surface*. NeurIPS.
- Brigo, D., Mercurio, F. (2002). *Lognormal-mixture dynamics and calibration
  to market volatility smiles*. IJTAF 5(4).
- Bayer, C., Friz, P., Gatheral, J. (2016). *Pricing under rough volatility*.
  Quantitative Finance 16(6).
- Breeden, D. T., Litzenberger, R. H. (1978). *Prices of state-contingent claims
  implicit in option prices*. Journal of Business 51(4).
- Kellerer, H. G. (1972). *Markov-Komposition und eine Anwendung auf
  Martingale*. Mathematische Annalen 198.
- Lee, R. W. (2004). *The moment formula for implied volatility at extreme
  strikes*. Mathematical Finance 14(3).
- Diebold, F. X., Mariano, R. S. (1995). *Comparing predictive accuracy*.
  Journal of Business & Economic Statistics 13(3).
