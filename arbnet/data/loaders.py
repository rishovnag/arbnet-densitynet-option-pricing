"""NSE options data loader.

NSE publishes daily F&O bhavcopy files at https://www.nseindia.com/all-reports
(navigate to "Derivatives Daily Reports"). The format has evolved; a typical
schema includes:

    SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN, HIGH, LOW, CLOSE,
    SETTLE_PR, CONTRACTS, VAL_INLAKH, OPEN_INT, CHG_IN_OI, TIMESTAMP

For our purposes we extract: STRIKE_PR, EXPIRY_DT, OPTION_TYP ('CE'/'PE'),
SETTLE_PR (preferred over CLOSE to avoid trade-time biases), OPEN_INT, plus
TIMESTAMP (snapshot date).

This loader supports two formats:
  1. NSE legacy CSV (FO_BHAV*.csv, columns above).
  2. NSE new format (FO_*.csv with renamed columns; we autodetect).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class OptionsSnapshot:
    """A single-day options snapshot for one underlying."""

    snapshot_date: pd.Timestamp
    underlying: str
    spot: float
    risk_free_rate: float
    dividend_yield: float
    strikes: np.ndarray         # shape (n,)
    expiries: np.ndarray        # shape (n,) datetime64
    times_to_expiry: np.ndarray # shape (n,) in years (ACT/365)
    option_types: np.ndarray    # shape (n,) 'C' or 'P'
    prices: np.ndarray          # shape (n,)
    open_interest: np.ndarray   # shape (n,)
    bid: Optional[np.ndarray] = None
    ask: Optional[np.ndarray] = None
    implied_vol: Optional[np.ndarray] = None

    def call_subset(self) -> "OptionsSnapshot":
        mask = self.option_types == "C"
        return self._mask(mask)

    def put_subset(self) -> "OptionsSnapshot":
        mask = self.option_types == "P"
        return self._mask(mask)

    def _mask(self, mask: np.ndarray) -> "OptionsSnapshot":
        return OptionsSnapshot(
            snapshot_date=self.snapshot_date,
            underlying=self.underlying,
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            strikes=self.strikes[mask],
            expiries=self.expiries[mask],
            times_to_expiry=self.times_to_expiry[mask],
            option_types=self.option_types[mask],
            prices=self.prices[mask],
            open_interest=self.open_interest[mask],
            bid=self.bid[mask] if self.bid is not None else None,
            ask=self.ask[mask] if self.ask is not None else None,
            implied_vol=self.implied_vol[mask] if self.implied_vol is not None else None,
        )

    def __len__(self) -> int:
        return len(self.strikes)


def _detect_format(df: pd.DataFrame) -> str:
    cols = set(c.upper() for c in df.columns)
    if {"STRIKE_PR", "OPTION_TYP", "EXPIRY_DT"}.issubset(cols):
        return "legacy"
    if {"STRIKE_PRICE", "OPTN_TYPE", "EXPIRY_DATE"}.issubset(cols):
        return "new"
    raise ValueError(f"Unknown bhavcopy format; columns={df.columns.tolist()}")


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse NSE bhavcopy date columns robustly.

    Legacy bhavcopy dates are ``DD-Mon-YYYY`` (EXPIRY_DT title-cased like
    ``04-Jun-2020``; TIMESTAMP upper-cased like ``01-JAN-2020``). pandas'
    format inference latches onto ``%d-%B-%Y`` when the first value is a "May"
    date -- "May" is the one month whose name equals its own abbreviation --
    and then raises on the next "Jun". Parsing the (tiny) set of unique values
    with ``format="mixed"`` sidesteps that: each value is parsed independently.
    """
    s = s.astype(str).str.strip()
    uniq = pd.Index(s.unique())
    parsed = pd.to_datetime(uniq, format="mixed", dayfirst=True, errors="coerce")
    return s.map(dict(zip(uniq, parsed)))


def load_nse_options_csv(
    path: str,
    underlying: str = "NIFTY",
    spot: Optional[float] = None,
    risk_free_rate: float = 0.065,
    dividend_yield: float = 0.012,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> OptionsSnapshot:
    """Load one day's NSE F&O bhavcopy into an OptionsSnapshot.

    The spot must be supplied separately (NSE F&O bhavcopy does not always
    include the underlying spot directly); pass the Nifty 50 close for the same
    date. Risk-free rate and dividend yield default to commonly used Indian
    market figures (91-day T-bill ~6.5%, Nifty 50 ~1.2% yield) but should be
    replaced with date-matched values for production use.
    """
    df = pd.read_csv(path)
    # Some NSE bhavcopy exports include a trailing comma per row, which pandas
    # reads as an extra unnamed column. Drop any all-NaN trailing columns so
    # downstream format detection is not fooled.
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    fmt = _detect_format(df)
    if fmt == "legacy":
        col_strike = "STRIKE_PR"
        col_type = "OPTION_TYP"
        col_expiry = "EXPIRY_DT"
        col_settle = "SETTLE_PR" if "SETTLE_PR" in df.columns else "CLOSE"
        col_oi = "OPEN_INT"
        col_symbol = "SYMBOL"
        col_inst = "INSTRUMENT" if "INSTRUMENT" in df.columns else None
        col_ts = "TIMESTAMP"
    else:
        col_strike = "STRIKE_PRICE"
        col_type = "OPTN_TYPE"
        col_expiry = "EXPIRY_DATE"
        col_settle = "SETL_PRICE" if "SETL_PRICE" in df.columns else "CLOSE_PRICE"
        col_oi = "OPEN_INT"
        col_symbol = "TCKR_SYMB" if "TCKR_SYMB" in df.columns else "SYMBOL"
        col_inst = "FININSTRM_TP_CD" if "FININSTRM_TP_CD" in df.columns else None
        col_ts = "TRADE_DT" if "TRADE_DT" in df.columns else "TIMESTAMP"

    # Filter to options on the desired underlying
    df = df[df[col_symbol].str.upper() == underlying.upper()]
    if col_inst is not None:
        mask_options = df[col_inst].isin(["OPTIDX", "OPTSTK", "OPIX", "OPTI"])
        df = df[mask_options | df[col_type].isin(["CE", "PE"])]
    df = df[df[col_type].isin(["CE", "PE"])].copy()
    if df.empty:
        raise ValueError(f"No option rows found for {underlying} in {path}")

    expiry = _parse_dates(df[col_expiry])
    snapshot = snapshot_date or _parse_dates(df[col_ts]).iloc[0]
    tte = (expiry - snapshot).dt.days.astype(float) / 365.0

    option_types = np.where(df[col_type].str.upper() == "CE", "C", "P")

    if spot is None:
        # Heuristic: use ATM put-call parity inversion if both legs present
        spot = _estimate_spot_from_parity(df, col_strike, col_settle, col_type, col_expiry, snapshot, risk_free_rate, dividend_yield)

    snap = OptionsSnapshot(
        snapshot_date=snapshot,
        underlying=underlying,
        spot=float(spot),
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        strikes=df[col_strike].to_numpy(dtype=float),
        expiries=expiry.to_numpy(),
        times_to_expiry=tte.to_numpy(),
        option_types=option_types,
        prices=df[col_settle].to_numpy(dtype=float),
        open_interest=df[col_oi].to_numpy(dtype=float),
    )
    return snap


def _estimate_spot_from_parity(df, col_strike, col_settle, col_type, col_expiry, snapshot, r, q) -> float:
    """Estimate spot via put-call parity at the nearest *future* expiry.

    Parity:  C - P = S e^{-qT} - K e^{-rT}.

    Raises ``ValueError`` when parity cannot be evaluated (no future expiry, or
    no strike carrying both a call and a put). The previous "median strike"
    fallback silently returned a meaningless number and is intentionally gone:
    a day whose spot cannot be inferred should be surfaced, not fudged.
    """
    expiry_parsed = _parse_dates(df[col_expiry])
    expiries = expiry_parsed.dropna().unique()
    future = [e for e in expiries if pd.Timestamp(e) > snapshot]
    if not future:
        raise ValueError("put-call parity: no future expiry in bhavcopy")
    nearest_expiry = min(future, key=lambda e: pd.Timestamp(e) - snapshot)
    T = max((pd.Timestamp(nearest_expiry) - snapshot).days / 365.0, 1e-6)
    sub = df[expiry_parsed == nearest_expiry]
    # groupby().last() rather than set_index() so duplicate strikes don't raise.
    calls = sub[sub[col_type].str.upper() == "CE"].groupby(col_strike)[col_settle].last()
    puts = sub[sub[col_type].str.upper() == "PE"].groupby(col_strike)[col_settle].last()
    common = calls.index.intersection(puts.index)
    if len(common) == 0:
        raise ValueError("put-call parity: no strike with both a call and a put")
    parity_estimates = []
    for K in common:
        C, P = float(calls.loc[K]), float(puts.loc[K])
        # S = (C - P + K e^{-rT}) / e^{-qT}
        parity_estimates.append((C - P + float(K) * np.exp(-r * T)) / np.exp(-q * T))
    return float(np.median(parity_estimates))
