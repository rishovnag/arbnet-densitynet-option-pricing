# Datasets

Everything ArbNet needs is committed under [`data/`](data/) at the repository
root, so both studies run with no downloads. This document records what each
dataset is, its schema, where it came from, and how to refresh it.

The loaders live in `arbnet/data/real_data.py`; they auto-discover `data/`.
Override the location with the `ARBNET_DATA_ROOT` environment variable. Run
`python scripts/check_data.py` to validate the whole tree against the loaders.

```
data/
├── nse/
│   ├── fo_bhavcopy/         # 1359 daily F&O bhavcopies (one CSV per trading day)
│   └── nifty50_spot.csv     # official Nifty 50 daily close
├── rates/
│   ├── Auctions of 91-Day Government of India Treasury Bills.xlsx
│   └── NIFTY 50-yield-*.csv # per-year NIFTY 50 P/E, P/B, dividend yield
├── auxiliary/
│   ├── india_vix.csv
│   ├── usdinr.csv
│   └── fii_dii.csv
└── macro/
    ├── cpi.csv  wpi.csv  iip.csv
    └── india_macro_calendar_extended.csv
```

---

## 1. NSE Nifty 50 options — `nse/fo_bhavcopy/` (primary, training data)

Daily settlement prices, open interest and volume for Nifty 50 (and Bank Nifty)
index options, weekly and monthly expiries.

- **Source:** National Stock Exchange of India — F&O Bhavcopy,
  <https://www.nseindia.com/all-reports> → Derivatives Daily Reports.
- **Coverage:** 2019-01-01 → 2024-07-05, 1359 trading days — spanning the
  COVID-2020 crash and the post-2022 RBI tightening cycle.
- **Format:** one CSV per trading day, `fo_YYYYMMDD.csv`, NSE legacy schema.

Fields used (legacy schema): `SYMBOL` (filtered to `NIFTY` / `BANKNIFTY`),
`INSTRUMENT` (`OPTIDX` / `FUTIDX`), `EXPIRY_DT`, `STRIKE_PR`, `OPTION_TYP`
(`CE` / `PE`), `SETTLE_PR` (preferred over `CLOSE`), `OPEN_INT`, `TIMESTAMP`.
`arbnet.data.loaders.load_nse_options_csv` auto-detects the legacy vs. newer
UDiFF schema.

**Pre-filtering.** The raw bhavcopy lists *every* F&O instrument (~34k rows/day —
single-stock options and futures, all index families). The committed files are
filtered to the only symbols this project uses, **`NIFTY` and `BANKNIFTY`**,
which shrinks the dataset ~10× (≈4.9 GB → ≈430 MB) and changes nothing the
loaders see (they filter by symbol regardless). To recover the full
multi-asset bhavcopy, re-fetch with `scripts/prepare_data.py options`.

**Caveat.** Deep-OTM weekly options on NSE often carry stale settlement prices
after volatility spikes. The default `FilterConfig` drops contracts with
`SETTLE_PR < 0.05`, `|log(K/F)| > 0.5`, or that violate no-arbitrage bounds.

## 2. Nifty 50 spot — `nse/nifty50_spot.csv` (required)

Official daily closing level of the Nifty 50 index.

- **Source:** Yahoo Finance ticker `^NSEI` (equivalently NSE historical index
  data).
- **Schema:** `date, adj close, close, high, low, open, volume`.
- **Used for:** the spot `S` (authoritative; `load_snapshot` falls back to a
  futures-implied spot, then put-call parity, only when a date is missing), and
  the `realized_vol_1w` context feature.

## 3. Risk-free rate — `rates/Auctions of 91-Day…T-bills.xlsx` (required)

91-day Government of India Treasury-bill cut-off yields.

- **Source:** Reserve Bank of India — primary-auction results.
- **Used for:** the risk-free rate `r`, looked up as the most recent auction on
  or before each snapshot date (`rate_on`). Requires `openpyxl`; without it the
  loader falls back to a 6.5% constant.

## 4. Dividend yield — `rates/NIFTY 50-yield-*.csv` (required, low-impact)

Per-year NIFTY 50 index factsheet series (P/E, P/B, **dividend yield %**).

- **Source:** NSE indices (`niftyindices.com`).
- **Used for:** the dividend yield `q` (`div_yield_on`, as-of lookup). IV
  sensitivity to `q` is modest (< 30 bps).

## 5. Implied volatility (derived, no file)

Per-contract Black-Scholes implied volatility is inverted from market settle
prices by `arbnet.data.iv.implied_vol_newton`, a vectorised Newton inverter with
intrinsic-value and upper-bound guards. No download needed.

## 6. Auxiliary context features — `auxiliary/`

Fed to ArbNet's ICNN / monotone modules as a per-day context vector. ArbNet
remains arbitrage-free for any context, so these are safe to include.

| File | Contents | Source |
|---|---|---|
| `india_vix.csv` | India VIX daily close | NSE / Yahoo `^INDIAVIX` |
| `usdinr.csv` | USD/INR daily close | Yahoo `INR=X` |
| `fii_dii.csv` | daily FII / DII net cash-market flows (₹ crore) | niftytrader.in, back-filled from web.archive.org snapshots |

`fii_dii.csv` schema: `date, fii_net_cr, dii_net_cr` (NSE provisional figures).
Bank Nifty IV — a fourth auxiliary signal — is *not* a file: it is computed
on the fly from the `BANKNIFTY` options inside each bhavcopy
(`banknifty_atm_iv_on`).

## 7. Macro series and event calendar — `macro/`

| File | Contents | Source |
|---|---|---|
| `cpi.csv` | CPI All-India Combined, base 2012 (headline + CFPI, y-o-y %) | MoSPI |
| `wpi.csv` | WPI All Commodities, base 2011-12 (index + y-o-y %) | Office of the Economic Adviser, MoCI |
| `iip.csv` | IIP General Index, base 2011-12 (index + y-o-y %) | MoSPI |
| `india_macro_calendar_extended.csv` | RBI MPC dates, Union Budget, scheduled CPI/WPI/IIP releases — `date, event_type, severity, note` | RBI / PIB / MoSPI, hand-curated |

The monthly macro series are **lagged to their real publication dates** by the
pipeline (CPI +13 d, WPI +15 d, IIP +43 d from month-end) to avoid look-ahead
bias; the event calendar is pre-announced and used as-of.

## 8. Synthetic surfaces (no download)

For unit tests and the synthetic study the package ships two generators in
`arbnet/data/synthetic.py`:

- `SyntheticSurfaceGenerator` — fast, SSVI-parametric, arbitrage-free by
  construction.
- `RoughBergomiGenerator` — rough Bergomi Monte-Carlo; realistic short-dated
  smile, also the ground-truth DGP for the hedging-PnL simulations.

---

## Refreshing the data

`scripts/prepare_data.py` re-fetches the raw datasets into a `data/`-shaped
tree (it needs `requests`, `yfinance`, `tqdm` — `pip install -e ".[fetch]"`):

```bash
python scripts/prepare_data.py options --start 2019-01-01 --end 2024-12-31
python scripts/prepare_data.py spot         # Nifty 50 spot via yfinance
python scripts/prepare_data.py vix usdinr   # auxiliary series
python scripts/prepare_data.py all          # everything
```

NSE serves two bhavcopy schemas (legacy ZIP pre-mid-2024, UDiFF `.csv.gz`
after) and requires a primed browser-like session; the script handles both and
is idempotent (re-running only fetches missing days). The RBI T-bill series and
the dividend-yield factsheet have no stable CSV endpoint — refresh those
manually from <https://dbie.rbi.org.in> and
<https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-50>.

> Note: data fetched by `prepare_data.py` is **not** pre-filtered to
> NIFTY/BANKNIFTY — the loaders filter by symbol at read time, so the full
> bhavcopy works unchanged; it is simply larger on disk.
