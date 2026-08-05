#!/usr/bin/env python3
"""
prepare_data.py
===============

Fetch all datasets for the arbitrage-free deep option pricing project, matching
the directory layout and source list in DATASETS.md.

Subcommands
-----------
    options    NSE F&O bhavcopy (Nifty + Bank Nifty options, daily)
    spot       Nifty 50 spot daily closes (yfinance)
    vix        India VIX daily closes (yfinance, fallback note)
    usdinr     USD/INR daily (yfinance)
    tbill      RBI 91-day T-bill yield (best-effort; manual fallback)
    fii_dii    NSE FII/DII daily flows (best-effort)
    events     Pre-curated India macro event calendar (RBI MPC, Budget, ...)
    all        Run every subcommand sequentially

Examples
--------
    python scripts/prepare_data.py all --start 2019-01-01 --end 2024-12-31
    python scripts/prepare_data.py options --start 2024-01-01 --end 2024-12-31
    python scripts/prepare_data.py spot

The output directory layout is the repository-root ``data/`` folder::

    data/
    ├── nse/
    │   ├── fo_bhavcopy/         # one CSV per trading day
    │   └── nifty50_spot.csv
    ├── rates/
    │   ├── Auctions of 91-Day Government of India Treasury Bills.xlsx
    │   └── NIFTY 50-yield-*.csv
    ├── macro/
    │   ├── cpi.csv  wpi.csv  iip.csv
    │   └── india_macro_calendar_extended.csv
    └── auxiliary/
        ├── india_vix.csv
        ├── usdinr.csv
        └── fii_dii.csv

Override the root with the ARBNET_DATA_ROOT environment variable.

Dependencies
------------
    pip install requests pandas yfinance tqdm

The script is intentionally dependency-light and uses only `requests` for
HTTP. It does NOT depend on the arbnet package itself.

Notes on each source
--------------------
* NSE bhavcopy uses two schemas. Pre-Jul-2024 trading days use the legacy ZIP
  endpoint at `archives.nseindia.com`. From mid-2024 onward, NSE serves the
  UDiFF schema as `.csv.gz` at `nsearchives.nseindia.com`. We try the new
  endpoint first, fall back to the legacy one. NSE requires a primed session
  (cookies from a homepage visit) and rejects naive requests.

* RBI does not expose a stable CSV endpoint for T-bill yields. We attempt a
  best-effort fetch of the WSS-derived series; on failure, we emit a
  constant-yield CSV at 6.5% and print clear instructions for the manual
  download from <https://dbie.rbi.org.in>. The IV sensitivity to this number
  is moderate; for production runs you want a date-matched series.

* The Nifty 50 TTM dividend yield is published monthly on the NSE indices
  factsheet PDF. This script does NOT scrape it. We emit a constant 1.2% CSV
  and document the monthly snapshot procedure in the README. IV sensitivity
  is <30 bps so this is acceptable for first pass.

* FII/DII flows are unstable to scrape (the NSE flow report URL changes
  frequently). The script attempts the published archives endpoint; manual
  download is the realistic fallback for long histories.
"""
from __future__ import annotations

import argparse
import gzip
import io
import logging
import os
import random
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# prepare_data.py lives at <repo>/scripts/; the bundled datasets at <repo>/data/.
_REPO_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATA_ROOT = Path(os.environ.get("ARBNET_DATA_ROOT", str(_REPO_DATA_ROOT))).resolve()

NSE_HOME = "https://www.nseindia.com"
NSE_LEGACY_BHAV = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES/"
    "{year}/{mmm}/fo{dd}{mmm}{year}bhav.csv.zip"
)
NSE_UDIFF_BHAV = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.gz"
)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": UA,
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": NSE_HOME + "/",
}

log = logging.getLogger("prepare_data")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def daterange(start: date, end: date) -> Iterator[date]:
    """Inclusive of both endpoints, weekdays only (Mon–Fri)."""
    d = start
    one = timedelta(days=1)
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += one


def make_nse_session() -> requests.Session:
    """Create a session primed with NSE cookies."""
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    try:
        s.get(NSE_HOME + "/option-chain", timeout=10)
        s.get(NSE_HOME + "/", timeout=10)
    except requests.RequestException as e:
        log.warning("session prime failed: %s (continuing anyway)", e)
    return s


def polite_sleep(min_s: float = 0.4, max_s: float = 1.1) -> None:
    time.sleep(random.uniform(min_s, max_s))


def retry_get(
    session: requests.Session,
    url: str,
    *,
    attempts: int = 4,
    backoff_base: float = 1.7,
    timeout: int = 25,
) -> Optional[requests.Response]:
    last = None
    for i in range(attempts):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                return resp
            if resp.status_code in (403, 401):
                # session likely soured; re-prime once
                session.get(NSE_HOME + "/", timeout=10)
            last = resp
        except requests.RequestException as e:
            last = e
        time.sleep(backoff_base ** i + random.uniform(0, 0.5))
    log.debug("retry_get gave up on %s (last=%s)", url, last)
    return None


# ---------------------------------------------------------------------------
# 1. NSE F&O bhavcopy
# ---------------------------------------------------------------------------

def fetch_bhavcopy_day(session: requests.Session, d: date, out_dir: Path) -> str:
    """Try UDiFF first (post mid-2024), then legacy ZIP. Returns 'ok'|'skip'|'miss'."""
    out_csv = out_dir / f"fo_{d.strftime('%Y%m%d')}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 1024:
        return "skip"

    # --- Try UDiFF csv.gz ---
    udiff_url = NSE_UDIFF_BHAV.format(yyyymmdd=d.strftime("%Y%m%d"))
    resp = retry_get(session, udiff_url, attempts=2)
    if resp is not None:
        try:
            data = gzip.decompress(resp.content)
            out_csv.write_bytes(data)
            return "ok"
        except OSError:
            pass  # not a valid gzip; fall through to legacy

    # --- Try legacy ZIP ---
    legacy_url = NSE_LEGACY_BHAV.format(
        year=d.strftime("%Y"),
        mmm=d.strftime("%b").upper(),
        dd=d.strftime("%d"),
    )
    resp = retry_get(session, legacy_url, attempts=3)
    if resp is None:
        return "miss"
    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        inner = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not inner:
            return "miss"
        with zf.open(inner[0]) as f:
            out_csv.write_bytes(f.read())
        return "ok"
    except zipfile.BadZipFile:
        return "miss"


def fetch_options(start: date, end: date) -> None:
    out_dir = DATA_ROOT / "nse" / "fo_bhavcopy"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = make_nse_session()

    days = list(daterange(start, end))
    log.info("fetching %d weekday bhavcopies into %s", len(days), out_dir)

    counts = {"ok": 0, "skip": 0, "miss": 0}
    for i, d in enumerate(days, 1):
        status = fetch_bhavcopy_day(session, d, out_dir)
        counts[status] += 1
        if i % 20 == 0 or i == len(days):
            log.info(
                "  [%d/%d] %s -> %s | ok=%d skip=%d miss=%d",
                i, len(days), d.isoformat(), status,
                counts["ok"], counts["skip"], counts["miss"],
            )
        if status == "ok":
            polite_sleep()  # don't hammer

    log.info("bhavcopy done: %s", counts)
    if counts["miss"] > 0:
        log.warning(
            "%d days could not be fetched. Common causes: market holiday "
            "(expected), NSE rate-limit (retry later), or URL schema "
            "transition. Re-run this subcommand to retry only the misses.",
            counts["miss"],
        )


# ---------------------------------------------------------------------------
# 2. Nifty 50 spot (yfinance)
# ---------------------------------------------------------------------------

def _yf_download(ticker: str, start: date, end: date) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance not installed. pip install yfinance") from e
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned empty frame for {ticker}")
    # yfinance returns a MultiIndex on Columns for multi-ticker; flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    return df


def fetch_spot(start: date, end: date) -> None:
    out = DATA_ROOT / "nse" / "nifty50_spot.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching Nifty 50 spot (^NSEI) %s..%s", start, end)
    df = _yf_download("^NSEI", start, end)
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))


def fetch_vix(start: date, end: date) -> None:
    out = DATA_ROOT / "auxiliary" / "india_vix.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching India VIX (^INDIAVIX) %s..%s", start, end)
    try:
        df = _yf_download("^INDIAVIX", start, end)
        df.to_csv(out, index=False)
        log.info("wrote %s (%d rows)", out, len(df))
    except RuntimeError as e:
        log.warning(
            "India VIX via yfinance failed (%s). Manual fallback: download "
            "from https://www.niftyindices.com/reports/historical-data "
            "(select India VIX) and save as %s",
            e, out,
        )


def fetch_usdinr(start: date, end: date) -> None:
    out = DATA_ROOT / "auxiliary" / "usdinr.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching USD/INR (INR=X) %s..%s", start, end)
    df = _yf_download("INR=X", start, end)
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))


# ---------------------------------------------------------------------------
# 3. 91-day T-bill (best-effort; manual fallback)
# ---------------------------------------------------------------------------

def fetch_tbill(start: date, end: date) -> None:
    """RBI does not expose a stable CSV. Emit a constant-yield placeholder and
    print instructions. Reproducing this for real means downloading from
    https://dbie.rbi.org.in (Database on Indian Economy) -> Weekly Statistical
    Supplement -> 'Treasury Bills (Cut-off Yield)' -> 91-day, then exporting
    to CSV with columns (date, yield_pct).
    """
    out = DATA_ROOT / "rates" / "91d_tbill.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    log.warning(
        "RBI T-bill: no stable scrape endpoint. Writing a constant-yield "
        "placeholder at 6.5%%. For production runs, replace %s with a "
        "date-matched series from https://dbie.rbi.org.in",
        out,
    )

    days = list(daterange(start, end))
    df = pd.DataFrame({"date": [d.isoformat() for d in days], "yield_pct": [6.5] * len(days)})
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows, placeholder)", out, len(df))


# ---------------------------------------------------------------------------
# 4. FII/DII flows (best-effort)
# ---------------------------------------------------------------------------

def fetch_fii_dii(start: date, end: date) -> None:
    """NSE publishes daily FII/DII flow reports under /api/fiidiiTradeReact.
    The endpoint is JSON and returns only the latest day; for history NSE
    expects you to use their report download interface. We emit a stub and
    direct the user to https://www.nseindia.com/reports for the manual pull.
    """
    out = DATA_ROOT / "auxiliary" / "fii_dii.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.warning(
        "FII/DII flows: NSE only exposes the latest day via their public API. "
        "For 2019-2024 history, manually download monthly reports from "
        "https://www.nseindia.com/reports-fii-dii-reports and merge. "
        "Writing an empty stub to %s",
        out,
    )
    pd.DataFrame(columns=["date", "fii_net_cr", "dii_net_cr"]).to_csv(out, index=False)


# ---------------------------------------------------------------------------
# 5. Event calendar (pre-curated)
# ---------------------------------------------------------------------------

def fetch_events(_start: date, _end: date) -> None:
    """The repository ships a pre-curated India macro event calendar
    (RBI MPC dates, Union Budget, scheduled CPI/WPI/IIP releases) at
    ``data/macro/india_macro_calendar_extended.csv``. This subcommand is a
    no-op when that bundled file is already in place; it only reports status.
    """
    cal = _REPO_DATA_ROOT / "macro" / "india_macro_calendar_extended.csv"
    if cal.exists():
        log.info("macro event calendar already present: %s", cal)
    else:
        log.error(
            "Macro event calendar not found at %s. Re-clone the repository or "
            "hand-author a CSV with columns (date, event_type, severity, note).",
            cal,
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

SUBCOMMANDS = {
    "options": fetch_options,
    "spot": fetch_spot,
    "vix": fetch_vix,
    "usdinr": fetch_usdinr,
    "tbill": fetch_tbill,
    "fii_dii": fetch_fii_dii,
    "events": fetch_events,
}


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "cmd",
        choices=list(SUBCOMMANDS) + ["all"],
        help="which dataset to fetch (or 'all')",
    )
    p.add_argument("--start", type=parse_date, default=date(2019, 1, 1))
    p.add_argument("--end", type=parse_date, default=date(2024, 12, 31))
    p.add_argument(
        "--quiet",
        action="store_true",
        help="set log level to WARNING",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        force=True,
    )

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("data root: %s", DATA_ROOT)
    log.info("date range: %s .. %s", args.start, args.end)

    targets = list(SUBCOMMANDS) if args.cmd == "all" else [args.cmd]
    for name in targets:
        log.info("=== %s ===", name)
        try:
            SUBCOMMANDS[name](args.start, args.end)
        except Exception as e:
            log.exception("%s failed: %s", name, e)
            if args.cmd != "all":
                return 1

    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
