"""Diagnostic script: walk the bundled real-data tree and verify that every
file the package's loaders expect actually exists, parses, and aligns with
the schemas the code assumes.

This does NOT touch the network. It runs entirely on the data files shipped
in the repository-root ``data/`` directory.

Usage:
    python scripts/check_data.py

Exit code 0 = all checks pass; 1 = one or more checks failed.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from arbnet.data import (
    default_data_root,
    list_bhavcopy_dates,
    bhavcopy_path_for,
    load_snapshot,
    apply_quality_filters,
    FilterConfig,
    load_india_vix,
    load_usdinr,
    load_fii_dii,
    load_cpi,
    load_wpi,
    load_iip,
    load_macro_calendar,
    load_nifty_div_yield,
    load_tbill_yields,
)


PASS = "OK  "
FAIL = "FAIL"


def report(label: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print(f"Data root: {default_data_root()}")
    all_ok = True

    # --- 1. Bhavcopy index ---
    print("\n[1] NSE F&O bhavcopy")
    dates = list_bhavcopy_dates()
    all_ok &= report("bhavcopy index built", len(dates) > 0, f"n_files={len(dates)}")
    if dates:
        all_ok &= report(
            "date range",
            True,
            f"{dates[0].date()} .. {dates[-1].date()}",
        )
        # Try loading the most recent day
        test_date = dates[-1]
        try:
            snap = load_snapshot(test_date)
            n_options = len(snap)
            all_ok &= report(
                f"load_snapshot({test_date.date()})",
                n_options > 0,
                f"n_options={n_options}, spot={snap.spot:.1f}, r={snap.risk_free_rate:.4f}, q={snap.dividend_yield:.4f}",
            )
            # Quality filter pass
            filt, summ = apply_quality_filters(snap, FilterConfig())
            all_ok &= report(
                f"apply_quality_filters",
                summ["n_kept"] > 0,
                f"kept {summ['n_kept']}/{summ['n_initial']}",
            )
        except Exception as e:
            all_ok &= report(f"load_snapshot({test_date.date()})", False, str(e))
            traceback.print_exc()

    # --- 2. Risk-free rate ---
    print("\n[2] Risk-free rate (91-day T-bill auctions)")
    try:
        s = load_tbill_yields()
        all_ok &= report(
            "T-bill series",
            True,
            f"n={len(s)}" + (f", range {s.index[0].date()}..{s.index[-1].date()}" if len(s) else ""),
        )
    except Exception as e:
        all_ok &= report("T-bill series", False, str(e))

    # --- 3. Dividend yield ---
    print("\n[3] NIFTY 50 dividend yield")
    try:
        dy = load_nifty_div_yield()
        all_ok &= report(
            "div yield series",
            len(dy) > 0,
            f"n={len(dy)}" + (f", first={dy.iloc[0]:.4f}, last={dy.iloc[-1]:.4f}" if len(dy) else ""),
        )
    except Exception as e:
        all_ok &= report("div yield series", False, str(e))

    # --- 4. Auxiliary series ---
    print("\n[4] Auxiliary series (VIX / USDINR / FII-DII)")
    try:
        vix = load_india_vix()
        all_ok &= report("india_vix.csv", len(vix) > 0, f"n={len(vix)}")
    except Exception as e:
        all_ok &= report("india_vix.csv", False, str(e))
    try:
        inr = load_usdinr()
        all_ok &= report("usdinr.csv", len(inr) > 0, f"n={len(inr)}")
    except Exception as e:
        all_ok &= report("usdinr.csv", False, str(e))
    try:
        fd = load_fii_dii()
        ok = ({"date", "fii_net_cr", "dii_net_cr"}.issubset(set(fd.columns))
              and len(fd) > 0)
        all_ok &= report("fii_dii.csv", ok, f"n_rows={len(fd)}")
    except Exception as e:
        all_ok &= report("fii_dii.csv", False, str(e))

    # --- 5. Macro ---
    print("\n[5] Macro (CPI / WPI / IIP / calendar)")
    for name, fn in [("cpi.csv", load_cpi), ("wpi.csv", load_wpi), ("iip.csv", load_iip)]:
        try:
            df = fn()
            all_ok &= report(name, len(df) > 0, f"n={len(df)}")
        except Exception as e:
            all_ok &= report(name, False, str(e))
    try:
        cal = load_macro_calendar()
        all_ok &= report("macro calendar", len(cal) > 0, f"n_events={len(cal)}")
    except Exception as e:
        all_ok &= report("macro calendar", False, str(e))

    # --- Summary ---
    print()
    if all_ok:
        print("=== ALL DATA CHECKS PASSED ===")
        return 0
    print("=== SOME DATA CHECKS FAILED ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
