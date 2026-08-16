"""Loaders for the bundled real Indian market data.

This module wires the on-disk datasets in the repository-root ``data/``
directory to the rest of the pipeline. Use it to exercise the package against
actual NSE bhavcopies instead of the synthetic surface generators in
``arbnet.data.synthetic``. See DATASETS.md for sources and schemas.

Data layout (auto-detected; override with the ARBNET_DATA_ROOT env var) ::

    data/
    ├── nse/
    │   ├── fo_bhavcopy/fo_YYYYMMDD.csv       # daily NSE F&O bhavcopy
    │   └── nifty50_spot.csv                  # official Nifty 50 close
    ├── rates/
    │   ├── NIFTY 50-yield-*.csv              # NIFTY 50 P/E, P/B, Div Yield%
    │   └── Auctions of 91-Day Government of India Treasury Bills.xlsx
    ├── auxiliary/
    │   ├── india_vix.csv
    │   ├── usdinr.csv
    │   └── fii_dii.csv
    └── macro/
        ├── cpi.csv, wpi.csv, iip.csv
        └── india_macro_calendar_extended.csv

All loaders are tolerant of missing files / partial data: they emit
warnings rather than raising, so the package can run with whatever subset
of the data is on disk.
"""
from __future__ import annotations

import glob
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .loaders import OptionsSnapshot, load_nse_options_csv


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback accounting (H2)
# ---------------------------------------------------------------------------
# Silent constant fallbacks for r/q made a fully de-rated run indistinguishable
# from a correct one. Every fallback is now counted (and logged); study
# scripts surface the totals in their output JSON so a run with a missing or
# unreadable rate file is visibly flagged.
_FALLBACKS = {
    "rate_default": 0,        # T-bill series empty/unreadable -> constant 6.5%
    "rate_prehistory": 0,     # date before first auction -> NaN (was: future yield)
    "div_default": 0,         # yield series empty/unreadable -> constant 1.2%
    "div_prehistory": 0,      # date before first yield row -> NaN (was: future value)
}


def reset_fallback_counters() -> None:
    for k in _FALLBACKS:
        _FALLBACKS[k] = 0


def fallback_report() -> dict:
    return dict(_FALLBACKS)


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------
# real_data.py lives at <repo>/arbnet/data/real_data.py; the bundled datasets
# live at <repo>/data/  -> three parents up, then "data".
_DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def default_data_root() -> Path:
    """Return the canonical on-disk data root used by the bundled loaders.

    Defaults to the ``data/`` directory at the repository root. Override via
    the ``ARBNET_DATA_ROOT`` environment variable.
    """
    env = os.environ.get("ARBNET_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_DATA_ROOT


# ---------------------------------------------------------------------------
# Bhavcopy index
# ---------------------------------------------------------------------------
def list_bhavcopy_dates(data_root: Optional[Path] = None) -> list[pd.Timestamp]:
    """Return sorted list of trading-day timestamps for which a bhavcopy
    file is on disk.
    """
    root = Path(data_root) if data_root else default_data_root()
    pat = str(root / "nse" / "fo_bhavcopy" / "fo_*.csv")
    out = []
    for p in glob.glob(pat):
        name = os.path.basename(p)
        # fo_YYYYMMDD.csv
        date_str = name[len("fo_"):-len(".csv")]
        try:
            out.append(pd.Timestamp(date_str))
        except ValueError:
            continue
    out.sort()
    return out


def bhavcopy_path_for(date: pd.Timestamp, data_root: Optional[Path] = None) -> Path:
    """Return the on-disk path of the bhavcopy file for ``date`` (raises if absent)."""
    root = Path(data_root) if data_root else default_data_root()
    p = root / "nse" / "fo_bhavcopy" / f"fo_{pd.Timestamp(date).strftime('%Y%m%d')}.csv"
    if not p.exists():
        raise FileNotFoundError(f"no bhavcopy on disk for {pd.Timestamp(date).date()}: {p}")
    return p


# ---------------------------------------------------------------------------
# Risk-free rate from the bundled T-bill auctions xlsx
# ---------------------------------------------------------------------------
def load_tbill_yields(data_root: Optional[Path] = None) -> pd.Series:
    """Load the bundled 91-day T-bill cutoff yields as a Series indexed by date
    (yields in percent). Returns an empty series if openpyxl / the file is missing.
    """
    root = Path(data_root) if data_root else default_data_root()
    xlsx = root / "rates" / "Auctions of 91-Day Government of India Treasury Bills.xlsx"
    if not xlsx.exists():
        log.warning("T-bill xlsx not found at %s; returning empty series", xlsx)
        return pd.Series(dtype=float, name="tbill_91d_pct")
    try:
        # The RBI sheet has: title at row 1, blank, "₹ Crores" at row 3, blank,
        # a 3-row merged header (rows 5-7), column numbers (row 8), then data
        # interleaved with single-cell year-section headers (e.g. "2026-27").
        # openpyxl emits a benign "no default style" UserWarning on this file;
        # we suppress only that one.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Workbook contains no default style",
                category=UserWarning,
            )
            raw = pd.read_excel(xlsx, engine="openpyxl", header=None)
    except ImportError:
        warnings.warn(
            "openpyxl is required to read the T-bill xlsx. "
            "`pip install openpyxl` or replace with a CSV.",
            stacklevel=2,
        )
        return pd.Series(dtype=float, name="tbill_91d_pct")
    except Exception as e:
        log.warning("could not parse %s: %s", xlsx, e)
        return pd.Series(dtype=float, name="tbill_91d_pct")

    # Locate the header row that contains "Date of Auction" and the column
    # holding "Implicit Yield at Cut-off Price". Falling back to the known
    # layout (col 1 = auction date, col 13 = implicit yield) if the strings
    # have shifted.
    date_idx = None
    yld_idx = None
    for r in range(min(20, len(raw))):
        row_vals = [str(v).strip().lower() if isinstance(v, str) else "" for v in raw.iloc[r].values]
        for c, v in enumerate(row_vals):
            if date_idx is None and "date of auction" in v:
                date_idx = (r, c)
            if yld_idx is None and "implicit yield" in v and "cut-off" in v:
                yld_idx = (r, c)
        if date_idx is not None and yld_idx is not None:
            break
    if date_idx is None or yld_idx is None:
        date_col, yld_col, data_start = 1, 13, 8
    else:
        date_col = date_idx[1]
        yld_col = yld_idx[1]
        data_start = max(date_idx[0], yld_idx[0]) + 1
        # Skip the numeric row (e.g. "1,2,3,...") if present immediately after the header block.
        if data_start < len(raw):
            first = raw.iloc[data_start].values
            if all((isinstance(v, (int, float)) and not pd.isna(v)) or (isinstance(v, str) and v.strip().isdigit())
                   for v in first if v is not None and not (isinstance(v, float) and pd.isna(v))):
                data_start += 1

    # The date column has openpyxl-native datetime values for data rows and
    # short strings like "2026-27" for year-section dividers. pandas can't
    # infer a single format from that mix, so it falls back to dateutil and
    # warns; the conversion is still correct (section rows become NaT and are
    # dropped below). Silence just that informational warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format",
            category=UserWarning,
        )
        dates = pd.to_datetime(raw.iloc[data_start:, date_col], errors="coerce")
    yields = pd.to_numeric(raw.iloc[data_start:, yld_col], errors="coerce")
    s = pd.Series(yields.values, index=pd.DatetimeIndex(dates), name="tbill_91d_pct").dropna()
    s = s[~s.index.isna()].sort_index()
    return s


def rate_on(date: pd.Timestamp, data_root: Optional[Path] = None, default: float = 0.065) -> float:
    """Look up the 91-day T-bill yield (decimal) for the most recent auction
    on or before ``date``.

    H2 semantics: if the series is empty/unreadable, WARN and fall back to
    ``default`` (counted in ``fallback_report()``). For dates BEFORE the first
    auction, return NaN -- the old behaviour returned the earliest (future)
    yield, a look-ahead leak on boundary days; callers must skip NaN days.
    """
    s = load_tbill_yields(data_root)
    if s.empty:
        _FALLBACKS["rate_default"] += 1
        log.warning("rate_on(%s): T-bill series empty/unreadable -> constant "
                    "default %.4f (check openpyxl + the rates/ xlsx)", date, default)
        return float(default)
    try:
        # most recent auction <= date
        s2 = s[s.index <= pd.Timestamp(date)]
        if s2.empty:
            _FALLBACKS["rate_prehistory"] += 1
            log.warning("rate_on(%s): date precedes first T-bill auction (%s) "
                        "-> NaN (skip this day)", date, s.index[0].date())
            return float("nan")
        return float(s2.iloc[-1]) / 100.0
    except Exception:
        _FALLBACKS["rate_default"] += 1
        log.warning("rate_on(%s): lookup failed -> constant default %.4f", date, default)
        return float(default)


# ---------------------------------------------------------------------------
# NIFTY 50 dividend yield from the bundled "NIFTY 50-yield-*.csv" files
# ---------------------------------------------------------------------------
def load_nifty_div_yield(data_root: Optional[Path] = None) -> pd.Series:
    """Concatenate all bundled "NIFTY 50-yield-*.csv" files into a single
    Series of decimal dividend yields indexed by date.
    """
    root = Path(data_root) if data_root else default_data_root()
    files = sorted(glob.glob(str(root / "rates" / "NIFTY 50-yield-*.csv")))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception as e:
            log.warning("could not read %s: %s", f, e)
            continue
        df.columns = [str(c).strip() for c in df.columns]
        # Locate columns flexibly.
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        dy_col = next((c for c in df.columns if "div" in c.lower() and "yield" in c.lower()), None)
        if date_col is None or dy_col is None:
            continue
        # The NSE indiacharts export uses DD-MMM-YYYY with uppercase month
        # abbreviations (e.g. "18-MAY-2017"). Normalise case so %b matches in
        # the C locale, then try that format first; fall back to the dayfirst
        # parser only for rows that don't match (under a suppressed warning).
        raw_dates = df[date_col].astype(str).str.strip().str.title()
        date_s = pd.to_datetime(raw_dates, format="%d-%b-%Y", errors="coerce")
        if date_s.isna().any():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format",
                    category=UserWarning,
                )
                fallback = pd.to_datetime(
                    df[date_col].astype(str).str.strip(),
                    errors="coerce",
                    dayfirst=True,
                )
            date_s = date_s.fillna(fallback)
        sub = pd.DataFrame({
            "date": date_s,
            "div_yield_pct": pd.to_numeric(df[dy_col], errors="coerce"),
        }).dropna()
        frames.append(sub)
    if not frames:
        return pd.Series(dtype=float, name="div_yield")
    big = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
    return pd.Series(big["div_yield_pct"].to_numpy() / 100.0,
                     index=pd.DatetimeIndex(big["date"]),
                     name="div_yield")


def div_yield_on(date: pd.Timestamp, data_root: Optional[Path] = None, default: float = 0.012) -> float:
    """Most recent NIFTY 50 dividend yield on or before ``date``.

    H2 semantics as in :func:`rate_on`: WARN + constant fallback (counted) when
    the series is missing; NaN (never a future value) for pre-history dates.
    """
    s = load_nifty_div_yield(data_root)
    if s.empty:
        _FALLBACKS["div_default"] += 1
        log.warning("div_yield_on(%s): yield series empty/unreadable -> constant "
                    "default %.4f", date, default)
        return float(default)
    s2 = s[s.index <= pd.Timestamp(date)]
    if s2.empty:
        _FALLBACKS["div_prehistory"] += 1
        log.warning("div_yield_on(%s): date precedes first yield row (%s) "
                    "-> NaN (skip this day)", date, s.index[0].date())
        return float("nan")
    return float(s2.iloc[-1])


# ---------------------------------------------------------------------------
# Auxiliary CSVs (yfinance schema: date, adj close, close, high, low, open, volume)
# ---------------------------------------------------------------------------
def _load_yf_csv(path: Path, col: str = "close") -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float, name=path.stem)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.warning("could not read %s: %s", path, e)
        return pd.Series(dtype=float, name=path.stem)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns or col not in df.columns:
        log.warning("%s missing required columns (date,%s); got %s", path, col, df.columns.tolist())
        return pd.Series(dtype=float, name=path.stem)
    idx = pd.to_datetime(df["date"], errors="coerce")
    s = pd.to_numeric(df[col], errors="coerce")
    return pd.Series(s.to_numpy(), index=pd.DatetimeIndex(idx), name=path.stem).dropna().sort_index()


def load_india_vix(data_root: Optional[Path] = None) -> pd.Series:
    root = Path(data_root) if data_root else default_data_root()
    return _load_yf_csv(root / "auxiliary" / "india_vix.csv", col="close")


def load_usdinr(data_root: Optional[Path] = None) -> pd.Series:
    root = Path(data_root) if data_root else default_data_root()
    return _load_yf_csv(root / "auxiliary" / "usdinr.csv", col="close")


_EMPTY_FII_DII = ["date", "fii_net_cr", "dii_net_cr"]


def load_fii_dii(data_root: Optional[Path] = None) -> pd.DataFrame:
    """Load the bundled FII/DII daily net cash-market flow series.

    Reads ``auxiliary/fii_dii.csv`` (canonical schema ``date, fii_net_cr,
    dii_net_cr``; INR crore, NSE provisional figures). Returns an empty frame
    with that schema if the file is absent or unparseable.
    """
    root = Path(data_root) if data_root else default_data_root()
    p = root / "auxiliary" / "fii_dii.csv"
    if not p.exists():
        return pd.DataFrame(columns=_EMPTY_FII_DII)
    try:
        raw = pd.read_csv(p, comment="#")
    except Exception as e:
        log.warning("could not read %s: %s", p, e)
        return pd.DataFrame(columns=_EMPTY_FII_DII)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "date" not in raw.columns:
        return pd.DataFrame(columns=_EMPTY_FII_DII)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for col in ("fii_net_cr", "dii_net_cr"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else np.nan
    return raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Macro data (CPI / WPI / IIP / calendars)
# ---------------------------------------------------------------------------
def _load_macro_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        # The macro CSVs have several `# ...` comment header lines. Some of
        # those lines are wrapped in double quotes (so they survived a tool
        # that escaped embedded commas), e.g. `"# Severity: 1=low, 2=mid"`.
        # pandas' `comment="#"` only triggers on a literal leading `#`, so it
        # mis-parses the quoted variants. Strip both shapes up-front.
        from io import StringIO
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            lines = f.readlines()
        kept = []
        for ln in lines:
            stripped = ln.lstrip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith('"#') or stripped.startswith("'#"):
                continue
            kept.append(ln)
        df = pd.read_csv(StringIO("".join(kept)))
    except Exception as e:
        log.warning("could not read %s: %s", path, e)
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_cpi(data_root: Optional[Path] = None) -> pd.DataFrame:
    root = Path(data_root) if data_root else default_data_root()
    return _load_macro_csv(root / "macro" / "cpi.csv")


def load_wpi(data_root: Optional[Path] = None) -> pd.DataFrame:
    root = Path(data_root) if data_root else default_data_root()
    return _load_macro_csv(root / "macro" / "wpi.csv")


def load_iip(data_root: Optional[Path] = None) -> pd.DataFrame:
    root = Path(data_root) if data_root else default_data_root()
    return _load_macro_csv(root / "macro" / "iip.csv")


def load_macro_calendar(data_root: Optional[Path] = None) -> pd.DataFrame:
    """Load the India macro event calendar -- RBI MPC dates, Union Budget, and
    scheduled CPI/WPI/IIP releases -- with columns ``date, event_type,
    severity, note``.
    """
    root = Path(data_root) if data_root else default_data_root()
    p = root / "macro" / "india_macro_calendar_extended.csv"
    if p.exists():
        return _load_macro_csv(p)
    return pd.DataFrame(columns=["date", "event_type", "severity", "note"])


# ---------------------------------------------------------------------------
# Nifty 50 spot index (official daily close) + realised volatility
# ---------------------------------------------------------------------------
def load_nifty_spot(data_root: Optional[Path] = None) -> pd.Series:
    """Official Nifty 50 daily close from ``nse/nifty50_spot.csv`` (date-indexed)."""
    root = Path(data_root) if data_root else default_data_root()
    return _load_yf_csv(root / "nse" / "nifty50_spot.csv", col="close")


def nifty_spot_on(date: pd.Timestamp, data_root: Optional[Path] = None,
                  tol_days: int = 4) -> Optional[float]:
    """Official Nifty 50 close for ``date`` (exact trading day, else the most
    recent close within ``tol_days`` calendar days). Returns None if unavailable.
    """
    s = load_nifty_spot(data_root)
    if s.empty:
        return None
    date = pd.Timestamp(date).normalize()
    if date in s.index:
        return float(s.loc[date])
    window = s[(s.index <= date) & (s.index >= date - pd.Timedelta(days=tol_days))]
    return float(window.iloc[-1]) if not window.empty else None


def realized_vol(data_root: Optional[Path] = None, window: int = 5) -> pd.Series:
    """Annualised trailing realised volatility of the Nifty 50 from official
    daily closes: rolling std of log-returns over ``window`` days x sqrt(252).
    """
    s = load_nifty_spot(data_root)
    if len(s) < window + 1:
        return pd.Series(dtype=float, name="realized_vol")
    logret = np.log(s / s.shift(1))
    return (logret.rolling(window).std() * np.sqrt(252.0)).dropna().rename("realized_vol")


def _index_future_spot(path: str, snapshot: pd.Timestamp, r: float, q: float,
                       symbol: str = "NIFTY") -> Optional[float]:
    """Spot proxy from the nearest index future for ``symbol`` in a bhavcopy:
    S = F * exp(-(r - q) * T). Returns None if no usable future row is found.
    """
    try:
        df = pd.read_csv(path).dropna(axis=1, how="all")
    except Exception:
        return None
    df.columns = [str(c).strip().upper() for c in df.columns]
    settle = next((c for c in ("SETTLE_PR", "CLOSE", "SETL_PRICE", "CLOSE_PRICE")
                   if c in df.columns), None)
    if not {"INSTRUMENT", "SYMBOL", "EXPIRY_DT"}.issubset(df.columns) or settle is None:
        return None
    fut = df[(df["INSTRUMENT"].astype(str).str.upper() == "FUTIDX")
             & (df["SYMBOL"].astype(str).str.upper() == symbol.upper())].copy()
    if fut.empty:
        return None
    # format="mixed" parses each value independently -- avoids the pandas
    # "May" abbreviation-vs-full-name format-inference trap (see _parse_dates).
    fut["EXPIRY"] = pd.to_datetime(
        fut["EXPIRY_DT"].astype(str).str.strip(),
        format="mixed", dayfirst=True, errors="coerce",
    )
    fut = fut[fut["EXPIRY"] >= snapshot]
    if fut.empty:
        return None
    near = fut.loc[fut["EXPIRY"].idxmin()]
    T = max((pd.Timestamp(near["EXPIRY"]) - snapshot).days / 365.0, 1e-6)
    F = pd.to_numeric(pd.Series([near[settle]]), errors="coerce").iloc[0]
    if not np.isfinite(F) or F <= 0:
        return None
    return float(F) * float(np.exp(-(r - q) * T))


def _nifty_future_spot(path: str, snapshot: pd.Timestamp,
                       r: float, q: float) -> Optional[float]:
    """Spot proxy from the nearest NIFTY index future (see ``_index_future_spot``)."""
    return _index_future_spot(path, snapshot, r, q, symbol="NIFTY")


def banknifty_atm_iv_on(date: pd.Timestamp, data_root: Optional[Path] = None) -> float:
    """ATM Black-Scholes implied vol of the nearest-month Bank Nifty index call
    for ``date``, inverted from the bhavcopy.

    Bank Nifty is the second index whose options ship inside every NSE F&O
    bhavcopy; its ATM IV is a useful cross-sectional "vol of vol" context
    signal (DATASETS.md s6). Returns NaN when it cannot be computed.
    """
    import torch  # local import: keeps the data layer importable without eager torch
    from .iv import implied_vol_newton

    date = pd.Timestamp(date).normalize()
    try:
        path = bhavcopy_path_for(date, data_root)
    except FileNotFoundError:
        return float("nan")
    r = rate_on(date, data_root)
    q = div_yield_on(date, data_root)
    bn_spot = _index_future_spot(str(path), date, r, q, symbol="BANKNIFTY")
    if bn_spot is None or bn_spot <= 0:
        return float("nan")
    try:
        snap = load_nse_options_csv(
            str(path), underlying="BANKNIFTY", spot=bn_spot,
            risk_free_rate=r, dividend_yield=q, snapshot_date=date,
        )
    except ValueError:
        return float("nan")
    calls = snap.call_subset()
    if len(calls) == 0:
        return float("nan")
    T = np.asarray(calls.times_to_expiry, dtype=float)
    K = np.asarray(calls.strikes, dtype=float)
    P = np.asarray(calls.prices, dtype=float)
    ok = (np.isfinite(T) & np.isfinite(K) & np.isfinite(P)
          & (P > 0.05) & (T >= 7 / 365.0) & (T <= 90 / 365.0))
    if not ok.any():
        return float("nan")
    T, K, P = T[ok], K[ok], P[ok]
    t_near = float(np.unique(T).min())
    m = T == t_near
    Km, Pm = K[m], P[m]
    j = int(np.argmin(np.abs(Km - bn_spot)))   # ATM strike at the nearest expiry
    iv = implied_vol_newton(
        torch.tensor([float(Pm[j])]), torch.tensor([float(bn_spot)]),
        torch.tensor([float(Km[j])]), torch.tensor([t_near]),
        torch.tensor([float(r)]), torch.tensor([float(q)]),
    )
    val = float(iv.item())
    return val if (np.isfinite(val) and 0.01 <= val <= 3.0) else float("nan")


class SnapshotQualityError(ValueError):
    """Raised when a bhavcopy day cannot yield a trustworthy snapshot
    (no valid strikes, unresolvable spot, or a spot inconsistent with the
    traded data). Callers should skip such days and log the reason.
    """


# ---------------------------------------------------------------------------
# High-level: build a snapshot for a given trading day using all bundled data
# ---------------------------------------------------------------------------
def load_snapshot(
    date: pd.Timestamp,
    data_root: Optional[Path] = None,
    underlying: str = "NIFTY",
    spot: Optional[float] = None,
    rate: Optional[float] = None,
    div_yield: Optional[float] = None,
) -> OptionsSnapshot:
    """Build an OptionsSnapshot for ``date`` from the bundled data.

    Risk-free rate comes from the 91-day T-bill auctions, dividend yield from
    the NIFTY 50-yield CSVs. The spot is resolved in order of preference:

      1. caller-supplied ``spot``;
      2. official Nifty 50 close (``nse/nifty50_spot.csv``) -- authoritative;
      3. nearest NIFTY index future in the bhavcopy, discounted to spot;
      4. put-call parity at the nearest expiry.

    Raises ``SnapshotQualityError`` when the day cannot yield a trustworthy
    snapshot -- corrupted days are surfaced to the caller, never silently
    fudged (the old median-strike fallback has been removed).
    """
    date = pd.Timestamp(date).normalize()
    path = bhavcopy_path_for(date, data_root)
    r = float(rate) if rate is not None else rate_on(date, data_root)
    q = float(div_yield) if div_yield is not None else div_yield_on(date, data_root)

    resolved_spot, spot_source = (
        (float(spot), "caller") if spot is not None else (None, None)
    )
    if resolved_spot is None:
        official = nifty_spot_on(date, data_root)
        if official is not None and official > 0:
            resolved_spot, spot_source = official, "official_close"
    future_spot = _nifty_future_spot(str(path), date, r, q)
    if resolved_spot is None and future_spot is not None and future_spot > 0:
        resolved_spot, spot_source = future_spot, "nifty_future"

    # Cross-check the official close against the futures-implied spot: a near
    # future sits within ~1% of spot, so a >8% gap signals a corrupt print.
    if (spot_source == "official_close" and future_spot is not None
            and future_spot > 0):
        gap = abs(resolved_spot / future_spot - 1.0)
        if gap > 0.08:
            raise SnapshotQualityError(
                f"{date.date()}: official close {resolved_spot:.1f} disagrees "
                f"with futures-implied spot {future_spot:.1f} by {gap:.1%}"
            )

    try:
        snap = load_nse_options_csv(
            str(path), underlying=underlying, spot=resolved_spot,
            risk_free_rate=r, dividend_yield=q, snapshot_date=date,
        )
    except ValueError as e:
        raise SnapshotQualityError(f"{date.date()}: {e}") from e
    if spot_source is None:
        spot_source = "parity"

    # Data-quality gate: spot must sit inside the traded strike range.
    strikes = np.asarray(snap.strikes, dtype=float)
    strikes = strikes[np.isfinite(strikes) & (strikes > 0)]
    if strikes.size == 0:
        raise SnapshotQualityError(f"{date.date()}: bhavcopy has no valid strikes")
    k_lo, k_hi = float(strikes.min()), float(strikes.max())
    if not (k_lo <= float(snap.spot) <= k_hi):
        raise SnapshotQualityError(
            f"{date.date()}: resolved spot {snap.spot:.1f} ({spot_source}) "
            f"outside traded strike range [{k_lo:.0f}, {k_hi:.0f}]"
        )
    log.debug("%s: spot %.2f via %s", date.date(), snap.spot, spot_source)
    return snap


def build_context_for(
    date: pd.Timestamp,
    data_root: Optional[Path] = None,
) -> dict:
    """Build a scalar context dict for ``date`` from the bundled auxiliary
    series (India VIX, USD/INR, FII/DII net flows). Missing series contribute
    NaN; downstream feature building can drop or impute.
    """
    date = pd.Timestamp(date).normalize()
    vix = load_india_vix(data_root)
    inr = load_usdinr(data_root)
    fd = load_fii_dii(data_root)

    def _asof(s: pd.Series) -> float:
        if s.empty:
            return float("nan")
        s2 = s[s.index <= date]
        return float(s2.iloc[-1]) if not s2.empty else float("nan")

    ctx = {
        "india_vix": _asof(vix),
        "usdinr": _asof(inr),
    }
    if not fd.empty:
        sub = fd[fd["date"] <= date]
        ctx["fii_net_cr"] = float(sub["fii_net_cr"].iloc[-1]) if not sub.empty else float("nan")
        ctx["dii_net_cr"] = float(sub["dii_net_cr"].iloc[-1]) if not sub.empty else float("nan")
    else:
        ctx["fii_net_cr"] = float("nan")
        ctx["dii_net_cr"] = float("nan")
    return ctx


__all__ = [
    "default_data_root",
    "list_bhavcopy_dates",
    "bhavcopy_path_for",
    "load_tbill_yields",
    "rate_on",
    "load_nifty_div_yield",
    "div_yield_on",
    "load_nifty_spot",
    "nifty_spot_on",
    "realized_vol",
    "load_india_vix",
    "load_usdinr",
    "load_fii_dii",
    "banknifty_atm_iv_on",
    "load_cpi",
    "load_wpi",
    "load_iip",
    "load_macro_calendar",
    "load_snapshot",
    "build_context_for",
    "SnapshotQualityError",
]
