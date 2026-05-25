#!/usr/bin/env python3
"""Walk-forward training + evaluation on the bundled real NSE Nifty 50 options data.

Real-data counterpart of ``scripts/run_study.py`` (synthetic rough-Bergomi
surfaces). It reproduces every stage of that study, 1:1, on actual NSE F&O
bhavcopy snapshots, and it consumes *every* dataset shipped in the
repository-root ``data/`` directory:

    nse/fo_bhavcopy/fo_*.csv ........ NIFTY option quotes -> training data;
                                      NIFTY future -> spot fallback;
                                      BANKNIFTY options -> context: Bank Nifty IV
    nse/nifty50_spot.csv ............ official Nifty 50 close -> spot S, and
                                      -> context feature: 1-week realised vol
    rates/...91-Day...T-bills.xlsx .. risk-free rate r          (per snapshot)
    rates/NIFTY 50-yield-*.csv ...... dividend yield q          (per snapshot)
    auxiliary/india_vix.csv ......... context feature: India VIX
    auxiliary/usdinr.csv ............ context feature: USD/INR
    auxiliary/fii_dii.csv ........... context feature: FII / DII net flows
    macro/cpi.csv ................... context feature: CPI headline inflation
    macro/wpi.csv ................... context feature: WPI inflation
    macro/iip.csv ................... context feature: IIP growth
    macro/india_macro_calendar_*.csv  context feature: days to next macro event

The auxiliary + macro series are assembled into a per-day context vector,
z-score normalised, and fed to ArbNet's ICNN / monotone modules via the
``context`` input (DATASETS.md s6). ArbNet stays static-arbitrage-free for
every context vector by construction, so this is safe. Macro prints are
lagged to their real publication dates (CPI +13d, WPI +15d, IIP +43d from
month-end) to avoid look-ahead bias; the event calendar is pre-announced so
it is used as-of. A macro feature is only active once at least one lagged
print falls inside the run window -- a window confined to early 2019 (before
the first release) will not see CPI/WPI/IIP; start from ~2019-03 onward.

Per trading day the pipeline is:

    bhavcopy -> load_snapshot (spot: official close -> nearest future ->
                               put-call parity; r from T-bills; q from yields;
                               corrupted days are skipped, never fudged)
             -> call_subset -> apply_quality_filters -> implied_vol_newton
             -> build_features (+ context) -> train_pricer
             -> evaluate: price RMSE, IV RMSE, static-arbitrage report
             -> delta-hedge PnL (std, CVaR95; guarded against degenerate vol)
    then:    per-model aggregate with bootstrap 95% CIs, a paired t-test, and
             HAC-corrected Diebold-Mariano tests; plus a data-quality report.

Claim scope: ArbNet is architecturally free of *butterfly* and *calendar*
static arbitrage (Theorem 4.1) -- reported under the guaranteed_* fields. The
Roger Lee wing/tail bound is monitored (monitored_tail_*) but NOT
architecturally enforced; tail violations are a diagnostic, not a breach.

Results are written to results/nse_study.json.

Examples
--------
    # Full study -- every available bhavcopy (~1359 days; the default).
    # This is hours on CPU; subsample with --stride for a faster pass.
    python scripts/train_nse.py --models arbnet ackerer bs --n_epochs 150

    # Faster: every 10th trading day across the full range
    python scripts/train_nse.py --stride 10 --models arbnet ackerer bs

    # Quick check: first 20 trading days only
    python scripts/train_nse.py --max_days 20 --models arbnet ackerer bs

    # Ablation: disable the context features (matches run_study.py exactly)
    python scripts/train_nse.py --no_context
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbnet.data import (
    list_bhavcopy_dates,
    load_snapshot,
    apply_quality_filters,
    FilterConfig,
    build_features,
    load_tbill_yields,
    load_nifty_div_yield,
    load_nifty_spot,
    realized_vol,
    load_india_vix,
    load_usdinr,
    load_fii_dii,
    banknifty_atm_iv_on,
    load_cpi,
    load_wpi,
    load_iip,
    load_macro_calendar,
)
from arbnet.data.iv import implied_vol_newton
from arbnet.models import ArbNet, AckererSoftPenaltyNet, BlackScholesPricer, RoughBergomiSimulator
from arbnet.utils import default_arbnet_config, RunConfig, set_seed
from arbnet.train import train_pricer
from arbnet.arbitrage import static_arbitrage_report
from arbnet.eval import rmse, hedge_pnl_stats, bootstrap_ci, diebold_mariano
from arbnet.hedging import hedge_pnl_delta


# =============================================================================
# Context features (auxiliary + macro data)
# =============================================================================
# Publication lag, in days after the month-end reference date, per macro
# series -- a print only becomes a usable feature once it would actually have
# been released (guards against look-ahead bias):
#   CPI ~12th of M+1, WPI ~14th of M+1, IIP ~12th of M+2.
CPI_LAG_DAYS = 13
WPI_LAG_DAYS = 15
IIP_LAG_DAYS = 43


def _monthly_series(df: pd.DataFrame, col: str, lag_days: int) -> pd.Series:
    """Turn a monthly macro DataFrame into a date-indexed Series, with the
    index shifted forward by ``lag_days`` so a print is only visible after it
    would actually have been published.
    """
    if df is None or df.empty or "date" not in df.columns or col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.Series(
        pd.to_numeric(df[col], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(df["date"]) + pd.Timedelta(days=lag_days),
    ).dropna().sort_index()
    return s


def _asof(series: pd.Series, date: pd.Timestamp) -> float:
    """Most recent value at or before ``date`` (NaN if none)."""
    if series is None or len(series) == 0:
        return float("nan")
    s2 = series[series.index <= date]
    return float(s2.iloc[-1]) if len(s2) else float("nan")


def _fd_asof(fd: pd.DataFrame, date: pd.Timestamp, col: str) -> float:
    if fd is None or fd.empty or "date" not in fd.columns or col not in fd.columns:
        return float("nan")
    sub = fd[fd["date"] <= date]
    return float(sub[col].iloc[-1]) if len(sub) else float("nan")


def _days_to_event(cal: pd.DataFrame, date: pd.Timestamp, cap: float = 60.0) -> float:
    """Calendar days until the next scheduled macro event (RBI MPC, Budget,
    CPI/WPI/IIP release). The calendar is pre-announced, so this is look-ahead
    safe. Clamped to [0, cap].
    """
    if cal is None or cal.empty or "date" not in cal.columns:
        return float("nan")
    future = cal["date"][cal["date"] >= date]
    if len(future) == 0:
        return cap
    return float(min((future.min() - date).days, cap))


def build_context_frame(dates: List[pd.Timestamp]) -> tuple:
    """Assemble the per-day context matrix from every auxiliary/macro file.

    Returns (normalised DataFrame indexed by 'YYYY-MM-DD', feature_names,
    source_report). Columns that are entirely missing over the selected window
    (e.g. the empty fii_dii.csv stub) are dropped automatically and reappear
    once the underlying file is populated.
    """
    vix = load_india_vix()
    inr = load_usdinr()
    fd = load_fii_dii()
    rv = realized_vol(window=5)   # 1-week realised vol from official Nifty closes
    cpi = _monthly_series(load_cpi(), "headline_inflation_pct", CPI_LAG_DAYS)
    wpi = _monthly_series(load_wpi(), "wpi_yoy_pct", WPI_LAG_DAYS)
    iip = _monthly_series(load_iip(), "iip_yoy_pct", IIP_LAG_DAYS)
    cal = load_macro_calendar()

    rows = []
    for d in dates:
        rows.append({
            "india_vix":           _asof(vix, d),
            "banknifty_iv":        banknifty_atm_iv_on(d),   # from BANKNIFTY options in the bhavcopy
            "realized_vol_1w":     _asof(rv, d),
            "usdinr":              _asof(inr, d),
            "fii_net_cr":          _fd_asof(fd, d, "fii_net_cr"),
            "dii_net_cr":          _fd_asof(fd, d, "dii_net_cr"),
            "cpi_inflation":       _asof(cpi, d),
            "wpi_inflation":       _asof(wpi, d),
            "iip_growth":          _asof(iip, d),
            "days_to_macro_event": _days_to_event(cal, d),
        })
    raw = pd.DataFrame(rows, index=[str(d.date()) for d in dates])

    source_report = {
        "india_vix_rows": int(len(vix)),
        "banknifty_iv_days": int(raw["banknifty_iv"].notna().sum()),
        "realized_vol_rows": int(len(rv)),
        "usdinr_rows": int(len(inr)),
        "fii_dii_rows": int(len(fd)),
        "cpi_rows": int(len(cpi)),
        "wpi_rows": int(len(wpi)),
        "iip_rows": int(len(iip)),
        "macro_calendar_events": int(len(cal)),
    }

    # Drop features with no data over this window (e.g. empty FII/DII stub).
    raw = raw.dropna(axis=1, how="all")
    if raw.shape[1] == 0:
        return raw, [], source_report

    # Impute: forward-fill across days, then fill any remainder with the
    # column median (a look-ahead-neutral constant).
    filled = raw.ffill()
    filled = filled.fillna(filled.median(numeric_only=True))
    filled = filled.fillna(0.0)

    # z-score normalise (the raw scales -- VIX~15, USDINR~80, IIP~+-50 -- differ
    # by orders of magnitude and would otherwise ill-condition the network).
    mu = filled.mean()
    sd = filled.std(ddof=0).replace(0.0, 1.0)
    norm = (filled - mu) / sd
    return norm, list(norm.columns), source_report


# =============================================================================
# Evaluation helpers
# =============================================================================
def _grid_eval(model, K_grid, T_grid, S0, r, q, ctx_vec: Optional[np.ndarray]):
    """Evaluate a model on a (T, K) grid in float64; return (C_grid, w_grid).

    The arbitrage report differences grid prices and counts a violation past a
    fixed 1e-7 tolerance. That tolerance is only "one decade above noise" in
    *double* precision: in float32, Nifty-scale call prices (thousands of INR,
    and deep-ITM legs worth more) carry ~1e-4 differencing noise, which
    manufactures hundreds of spurious butterfly/calendar violations on steep
    days. So the grid is evaluated in float64 -- ArbNet's architectural
    convexity/monotonicity then shows exactly, as it should.
    """
    n_T, n_K = T_grid.shape[0], K_grid.shape[0]
    Tm, Km = torch.meshgrid(T_grid.double(), K_grid.double(), indexing="ij")
    Tf, Kf = Tm.flatten(), Km.flatten()
    S_b = torch.full_like(Tf, float(S0))
    r_b = torch.full_like(Tf, float(r))
    q_b = torch.full_like(Tf, float(q))
    ctx_b = None
    if ctx_vec is not None:
        ctx_b = torch.tensor(ctx_vec, dtype=torch.float64).unsqueeze(0).expand(Tf.shape[0], -1)
    model.double()
    try:
        with torch.no_grad():
            out = model(Kf, Tf, S_b, r_b, q_b, ctx_b)
            if "w" in out and out["w"].abs().max() > 0:
                w = out["w"]
            else:
                iv = implied_vol_newton(out["price"], S_b, Kf, Tf, r_b, q_b)
                w = (iv * iv * Tf).nan_to_num(nan=0.0)
            price = out["price"]
    finally:
        model.float()   # restore: train / price-RMSE / hedge paths run in float32
    return price.view(n_T, n_K), w.view(n_T, n_K)


def _evaluate(model, feats: dict, snap, name: str, ctx_vec: Optional[np.ndarray]) -> Dict:
    """In-sample price/IV fit + static-arbitrage compliance for one trained model."""
    with torch.no_grad():
        out = model(feats["K"], feats["T"], feats["S"], feats["r"],
                    feats.get("q"), feats.get("context"))
    price_rmse = float(rmse(out["price"], feats["price"]))

    iv_model = implied_vol_newton(
        out["price"], feats["S"], feats["K"], feats["T"], feats["r"], feats.get("q"),
    )
    if "iv" in feats:
        valid = ~torch.isnan(iv_model) & ~torch.isnan(feats["iv"])
        iv_rmse = (
            float(((iv_model[valid] - feats["iv"][valid]) ** 2).mean().sqrt().item())
            if valid.any() else float("nan")
        )
    else:
        iv_rmse = float("nan")

    T_grid = torch.tensor(
        sorted({round(float(t), 6) for t in feats["T"].tolist()}), dtype=torch.float32,
    )
    K_grid = float(snap.spot) * torch.exp(torch.linspace(-0.35, 0.35, 41, dtype=torch.float32))
    k_grid = torch.log(K_grid / float(snap.spot))
    C_grid, w_grid = _grid_eval(
        model, K_grid, T_grid, snap.spot, snap.risk_free_rate, snap.dividend_yield, ctx_vec,
    )
    rep = static_arbitrage_report(
        C_grid=C_grid, w_grid=w_grid, K_grid=K_grid, T_grid=T_grid, k_grid=k_grid,
        r=float(snap.risk_free_rate), q=float(snap.dividend_yield),
    )
    return {
        "model": name,
        "price_rmse": price_rmse,
        "iv_rmse": iv_rmse,
        "butterfly_rate": rep.butterfly_rate,
        "calendar_rate": rep.calendar_rate,
        "tail_rate": rep.tail_rate,
        "total_rate": rep.total_rate,
    }


# A hedge run is only meaningful when the model's ATM implied vol is a sane
# number; outside this band the hedge ratio is garbage and the PnL tail
# statistics (CVaR95) become degenerate -- we record NaN instead of fabricating.
_SIGMA_MIN, _SIGMA_MAX = 0.01, 3.0
_NAN_HEDGE = {"hedge_std": float("nan"), "hedge_cvar95": float("nan")}


def _hedge_eval(model, snap, ctx_vec: Optional[np.ndarray], seed: int,
                T_h: float = 30 / 365.0, n_paths: int = 500) -> Dict:
    """One-step delta-hedge PnL of an ATM call, identical in form to
    run_study.py's _hedge_eval: paths are simulated from a fixed rough-Bergomi
    DGP, the hedge uses the model's own ATM implied vol.

    Guarded: if the model's ATM implied vol is NaN or implausible (outside
    [1%, 300%]), or the simulated paths are non-finite, the hedge metrics are
    returned as NaN rather than computed from a fabricated 20% vol. This stops
    a degenerate day from poisoning the aggregate hedge std / CVaR95.
    """
    atm = lambda: (torch.tensor([float(snap.spot)]), torch.tensor([T_h]),
                   torch.tensor([float(snap.spot)]),
                   torch.tensor([float(snap.risk_free_rate)]),
                   torch.tensor([float(snap.dividend_yield)]))
    ctx_row = (torch.tensor(ctx_vec, dtype=torch.float32).unsqueeze(0)
               if ctx_vec is not None else None)
    with torch.no_grad():
        if hasattr(model, "implied_vol"):
            K_a, T_a, S_a, r_a, q_a = atm()
            iv = model.implied_vol(K_a, T_a, S_a, r_a, q_a, context=ctx_row)
            sig = float(iv.item()) if not torch.isnan(iv).any() else float("nan")
        else:
            out_atm = model(*atm())
            sig = (float(out_atm["iv"].item())
                   if "iv" in out_atm and not torch.isnan(out_atm["iv"]).any()
                   else float("nan"))
    # Guard: an unusable ATM vol -> NaN hedge metrics (do not fabricate).
    if not (np.isfinite(sig) and _SIGMA_MIN <= sig <= _SIGMA_MAX):
        return dict(_NAN_HEDGE)

    sim = RoughBergomiSimulator(H=0.10, eta=1.5, rho=-0.7, xi0=0.04, dtype=torch.float64)
    t_grid = torch.linspace(0.0, T_h, 16, dtype=torch.float64)
    S_paths, _ = sim.simulate(n_paths, t_grid, S0=snap.spot,
                              r=snap.risk_free_rate, q=snap.dividend_yield, seed=seed)
    if not torch.isfinite(S_paths).all():
        return dict(_NAN_HEDGE)
    pnl = hedge_pnl_delta(
        S_paths.float(), torch.tensor(float(snap.spot)), torch.tensor(float(T_h)),
        torch.tensor(float(snap.risk_free_rate)), torch.tensor(sig),
        dt=float(T_h / 15), cost_bps=2.0,
    )
    if not torch.isfinite(pnl).all():
        return dict(_NAN_HEDGE)
    stats = hedge_pnl_stats(pnl)
    return {"hedge_std": float(stats.std), "hedge_cvar95": float(stats.cvar_95)}


# =============================================================================
# Per-day training
# =============================================================================
def _train_eval_day(name, feats, snap, ctx_vec, n_epochs, seed) -> Optional[Dict]:
    """Train one model on a day's features and return its evaluation record."""
    if name == "arbnet":
        ctx_dim = feats["context"].shape[1] if "context" in feats else 0
        model = ArbNet(default_arbnet_config(context_dim=ctx_dim))
        cfg = RunConfig(seed=seed, n_epochs=n_epochs, batch_size=64, lr=1e-2, lambda_iv=0.0)
        train_pricer(model, feats, cfg, verbose=False)
        rec = _evaluate(model, feats, snap, "arbnet", ctx_vec)
    elif name == "ackerer":
        model = AckererSoftPenaltyNet()  # context-free baseline
        cfg = RunConfig(seed=seed, n_epochs=n_epochs, batch_size=64, lr=1e-2,
                        lambda_iv=0.0, lambda_arb=1.0)
        K_pen = float(snap.spot) * torch.exp(torch.linspace(-0.3, 0.3, 21, dtype=torch.float32))
        T_pen = torch.tensor(
            sorted({round(float(t), 6) for t in feats["T"].tolist()}), dtype=torch.float32,
        )
        train_pricer(model, feats, cfg, soft_penalty_grid={"K_grid": K_pen, "T_grid": T_pen},
                     verbose=False)
        rec = _evaluate(model, feats, snap, "ackerer", None)
    elif name == "bs":
        model = BlackScholesPricer()
        cfg = RunConfig(seed=0, n_epochs=min(n_epochs, 80), batch_size=64, lr=2e-2,
                        lambda_iv=0.0)
        train_pricer(model, feats, cfg, verbose=False)
        rec = _evaluate(model, feats, snap, "bs", None)
    else:
        raise ValueError(f"unknown model {name}")
    rec.update(_hedge_eval(model, snap, ctx_vec if name == "arbnet" else None, seed))
    return rec


# =============================================================================
# Study driver
# =============================================================================
def run(dates, models, n_epochs, min_options, use_context, out_path, seed) -> Dict:
    set_seed(seed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- assemble the context matrix from all auxiliary/macro files ---------
    if use_context:
        ctx_norm, ctx_features, src_report = build_context_frame(dates)
        ctx_dim = len(ctx_features)
    else:
        ctx_norm, ctx_features, src_report, ctx_dim = pd.DataFrame(), [], {}, 0
    print(f"Context: {ctx_dim} feature(s) active{': ' + ', '.join(ctx_features) if ctx_features else ''}")
    if src_report:
        dropped = {"fii_net_cr", "dii_net_cr"} - set(ctx_features)
        if dropped and src_report.get("fii_dii_rows", 0) == 0:
            print("  (fii_dii.csv is an empty stub -> FII/DII features dropped; "
                  "populate it and they will be picked up automatically)")

    records: List[Dict] = []
    skipped: List[Dict] = []      # data-quality report: days dropped + why
    days_used = 0
    t0 = time.time()

    for di, date in enumerate(dates):
        try:
            snap_all = load_snapshot(date)              # calls + puts
        except Exception as e:
            reason = f"snapshot load failed: {e}"
            print(f"  [{date.date()}] SKIP -- {reason}")
            skipped.append({"date": str(date.date()), "reason": reason})
            continue
        snap = snap_all.call_subset()                   # ArbNet prices European calls
        filtered, _filt_summary = apply_quality_filters(snap, FilterConfig())
        if len(filtered) < min_options:
            reason = f"only {len(filtered)} calls after quality filter (< {min_options})"
            print(f"  [{date.date()}] SKIP -- {reason}")
            skipped.append({"date": str(date.date()), "reason": reason})
            continue

        n = len(filtered)
        feats_base = build_features(filtered)
        feats_base["iv"] = implied_vol_newton(
            feats_base["price"], feats_base["S"], feats_base["K"],
            feats_base["T"], feats_base["r"], feats_base.get("q"),
        )

        # Per-day context vector (z-scored) -> ArbNet only.
        ctx_vec = None
        feats_arb = feats_base
        if ctx_dim > 0:
            ctx_vec = ctx_norm.loc[str(date.date()), ctx_features].to_numpy(dtype=np.float32)
            feats_arb = dict(feats_base)
            feats_arb["context"] = torch.tensor(
                np.tile(ctx_vec, (n, 1)), dtype=torch.float32,
            )

        days_used += 1
        line = [f"  [{date.date()}] n={n:3d} S={snap.spot:9.1f} "
                f"r={snap.risk_free_rate:.4f} q={snap.dividend_yield:.4f} |"]
        for name in models:
            feats = feats_arb if name == "arbnet" else feats_base
            try:
                rec = _train_eval_day(name, feats, filtered, ctx_vec, n_epochs, seed + di)
            except Exception as e:
                print(f"  [{date.date()}] {name} FAILED: {e}")
                continue
            rec.update({
                "date": str(date.date()), "n_options": int(n),
                "spot": float(snap.spot),
                "risk_free_rate": float(snap.risk_free_rate),
                "dividend_yield": float(snap.dividend_yield),
            })
            records.append(rec)
            # arb = the architecturally guaranteed conditions (butterfly +
            # calendar); tail = the monitored-only Roger Lee diagnostic.
            arb_ok = "ok" if (rec["butterfly_rate"] == 0
                              and rec["calendar_rate"] == 0) else "VIOL"
            line.append(f" {name}:RMSE={rec['price_rmse']:7.1f} "
                         f"arb={arb_ok} tail={rec['tail_rate']:.2%} "
                         f"hStd={rec['hedge_std']:6.1f}")
        print("".join(line))

    wall = time.time() - t0

    # --- per-model aggregate ------------------------------------------------
    # Static-arbitrage compliance is split into the part ArbNet *guarantees*
    # architecturally -- butterfly (A1) + calendar (A2), Theorem 4.1 -- and the
    # Roger Lee wing/tail bound, which is only MONITORED, not enforced.
    agg: Dict[str, Dict] = {}
    for name in models:
        recs = [r for r in records if r["model"] == name]
        if not recs:
            continue
        rmse_arr = np.array([r["price_rmse"] for r in recs], dtype=float)
        hstd_arr = np.array([r["hedge_std"] for r in recs], dtype=float)
        hcv_arr = np.array([r["hedge_cvar95"] for r in recs], dtype=float)
        a = {
            "n_days": len(recs),
            "price_rmse_mean": float(np.mean(rmse_arr)),
            "price_rmse_std": float(np.std(rmse_arr)),
            "iv_rmse_mean": float(np.nanmean([r["iv_rmse"] for r in recs])),
            # architecturally guaranteed conditions:
            "guaranteed_butterfly_rate_mean": float(np.mean([r["butterfly_rate"] for r in recs])),
            "guaranteed_butterfly_rate_max": float(np.max([r["butterfly_rate"] for r in recs])),
            "guaranteed_calendar_rate_mean": float(np.mean([r["calendar_rate"] for r in recs])),
            "guaranteed_calendar_rate_max": float(np.max([r["calendar_rate"] for r in recs])),
            "guaranteed_violation_days": int(sum(
                1 for r in recs if r["butterfly_rate"] > 0 or r["calendar_rate"] > 0)),
            # monitored only -- NOT architecturally enforced:
            "monitored_tail_rate_mean": float(np.mean([r["tail_rate"] for r in recs])),
            "monitored_tail_rate_max": float(np.max([r["tail_rate"] for r in recs])),
            "hedge_std_mean": float(np.nanmean(hstd_arr)),
            "hedge_cvar95_mean": float(np.nanmean(hcv_arr)),
            "hedge_n_valid": int(np.isfinite(hstd_arr).sum()),
        }
        # Bootstrap 95% CIs on the per-model means (percentile bootstrap).
        if len(rmse_arr) >= 3:
            _, lo, hi = bootstrap_ci(rmse_arr, np.mean, seed=seed)
            a["price_rmse_ci95"] = [float(lo), float(hi)]
        hstd_v, hcv_v = hstd_arr[np.isfinite(hstd_arr)], hcv_arr[np.isfinite(hcv_arr)]
        if len(hstd_v) >= 3:
            _, lo, hi = bootstrap_ci(hstd_v, np.mean, seed=seed)
            a["hedge_std_ci95"] = [float(lo), float(hi)]
        if len(hcv_v) >= 3:
            _, lo, hi = bootstrap_ci(hcv_v, np.mean, seed=seed)
            a["hedge_cvar95_ci95"] = [float(lo), float(hi)]
        agg[name] = a

    # --- paired ArbNet-vs-Ackerer t-test on price RMSE (matched by day) -----
    paired = None
    arb = {r["date"]: r["price_rmse"] for r in records if r["model"] == "arbnet"}
    ack = {r["date"]: r["price_rmse"] for r in records if r["model"] == "ackerer"}
    diffs = np.array([arb[d] - ack[d] for d in arb if d in ack], dtype=float)
    if len(diffs) >= 2:
        mean_diff = float(diffs.mean())
        sd = float(diffs.std(ddof=1))
        nd = len(diffs)
        paired = {
            "n_pairs": nd,
            "mean_diff_arbnet_minus_ackerer": mean_diff,
            "std_diff": sd,
            "t_stat": mean_diff / (sd / math.sqrt(nd)) if sd > 0 else float("nan"),
        }

    # --- Diebold-Mariano tests (HAC-corrected: serial-correlation safe) ------
    # The plain paired t-test above ignores serial correlation across adjacent
    # trading days; DM with a Newey-West HAC variance corrects for it. We feed
    # per-day price RMSE as the per-observation loss, loss="absolute" so the
    # loss differential is rmse(model1) - rmse(model2).
    by_date: Dict[str, Dict[str, float]] = {}
    for r in records:
        by_date.setdefault(r["date"], {})[r["model"]] = r["price_rmse"]

    def _dm(m1: str, m2: str):
        common = sorted(d for d, v in by_date.items() if m1 in v and m2 in v)
        if len(common) < 5:
            return None
        e1 = np.array([by_date[d][m1] for d in common], dtype=float)
        e2 = np.array([by_date[d][m2] for d in common], dtype=float)
        res = diebold_mariano(e1, e2, loss="absolute")
        res["n"] = len(common)
        res["note"] = f"positive dm_stat => {m1} has higher price RMSE than {m2}"
        return res

    dm_tests: Dict[str, Optional[dict]] = {}
    if "arbnet" in models and "ackerer" in models:
        dm_tests["arbnet_vs_ackerer"] = _dm("arbnet", "ackerer")
    if "arbnet" in models and "bs" in models:
        dm_tests["arbnet_vs_bs"] = _dm("arbnet", "bs")

    summary = {
        "data": "NSE Nifty 50 F&O bhavcopy (real)",
        "claim_scope": (
            "ArbNet is architecturally free of butterfly (A1) and calendar (A2) "
            "static arbitrage for every weight vector and context (Theorem 4.1) "
            "-- see guaranteed_* fields. The Roger Lee wing/tail bound is "
            "MONITORED (monitored_tail_rate_*) but NOT architecturally enforced; "
            "treat tail violations as a diagnostic, not a breach of the guarantee."
        ),
        "params": {"n_days_requested": len(dates), "n_days_used": days_used,
                   "models": models, "n_epochs": n_epochs, "seed": seed,
                   "use_context": use_context},
        "context_features": ctx_features,
        "data_sources": src_report,
        "data_quality": {
            "n_requested": len(dates),
            "n_used": days_used,
            "n_skipped": len(skipped),
            "skipped_days": skipped,
        },
        "wall_time": wall,
        "n_records": len(records),
        "aggregate": agg,
        "paired_test": paired,
        "diebold_mariano": dm_tests,
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_path}  ({wall:.1f}s, {days_used} days used, "
          f"{len(skipped)} skipped, {len(records)} records)")
    return summary


def _parse_date(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(s, "%Y-%m-%d"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", type=_parse_date, default=None, help="first date YYYY-MM-DD")
    p.add_argument("--end", type=_parse_date, default=None, help="last date YYYY-MM-DD")
    p.add_argument("--stride", type=int, default=1, help="use every Nth trading day")
    p.add_argument("--max_days", type=int, default=0,
                   help="cap number of days (0 = all available bhavcopies, the default)")
    p.add_argument("--models", nargs="+", default=["arbnet", "ackerer", "bs"],
                   choices=["arbnet", "ackerer", "bs"])
    p.add_argument("--n_epochs", type=int, default=150)
    p.add_argument("--min_options", type=int, default=30,
                   help="skip days with fewer calls than this after filtering")
    p.add_argument("--no_context", action="store_true",
                   help="disable the auxiliary/macro context features (ablation)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/nse_study.json")
    args = p.parse_args()

    all_dates = list_bhavcopy_dates()
    if not all_dates:
        print("No bhavcopy files found under the data root. "
              "Run `python scripts/check_data.py` to diagnose.")
        sys.exit(1)
    dates = all_dates
    if args.start is not None:
        dates = [d for d in dates if d >= args.start]
    if args.end is not None:
        dates = [d for d in dates if d <= args.end]
    dates = dates[:: max(1, args.stride)]
    if args.max_days and args.max_days > 0:
        dates = dates[: args.max_days]
    if not dates:
        print("No trading days selected after applying --start/--end/--stride.")
        sys.exit(1)

    # Report which datasets are on disk and feeding the run.
    tbill_n = len(load_tbill_yields())
    divy_n = len(load_nifty_div_yield())
    print(f"NSE walk-forward: {len(dates)} trading days "
          f"({dates[0].date()} .. {dates[-1].date()}), models={args.models}")
    print(f"Rates: 91-day T-bill auctions ({tbill_n} rows) -> r;  "
          f"NIFTY 50 dividend yield ({divy_n} rows) -> q")

    summary = run(dates, args.models, args.n_epochs, args.min_options,
                  not args.no_context, args.out, args.seed)

    print("\n=== Aggregate (real NSE Nifty 50 data) ===")
    dq = summary["data_quality"]
    print(f"  days: {dq['n_used']} used / {dq['n_requested']} requested "
          f"({dq['n_skipped']} skipped for data quality)")
    for name, a in summary["aggregate"].items():
        print(f"  {name:8s}: n={a['n_days']:3d}  "
              f"RMSE={a['price_rmse_mean']:8.2f} +/- {a['price_rmse_std']:7.2f}  "
              f"IV={a['iv_rmse_mean']:.4f}  "
              f"guaranteed[butterfly={a['guaranteed_butterfly_rate_mean']:.3%} "
              f"calendar={a['guaranteed_calendar_rate_mean']:.3%} "
              f"viol_days={a['guaranteed_violation_days']}]  "
              f"monitored[tail={a['monitored_tail_rate_mean']:.3%}]  "
              f"hedgeStd={a['hedge_std_mean']:7.1f}  CVaR95={a['hedge_cvar95_mean']:8.1f}")
    pt = summary.get("paired_test")
    if pt:
        print(f"\nPaired ArbNet-Ackerer price RMSE: "
              f"delta={pt['mean_diff_arbnet_minus_ackerer']:+.2f}  "
              f"t={pt['t_stat']:.2f}  (n_pairs={pt['n_pairs']}, "
              f"uncorrected -- see Diebold-Mariano below)")
    for label, dm in (summary.get("diebold_mariano") or {}).items():
        if dm:
            print(f"Diebold-Mariano {label}: dm_stat={dm['dm_stat']:+.2f}  "
                  f"p={dm['p_value']:.4f}  (n={dm['n']}, HAC-corrected)")


if __name__ == "__main__":
    main()
