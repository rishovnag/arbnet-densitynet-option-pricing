#!/usr/bin/env python3
"""Regenerate the five manuscript figures from the ArbNet study results.

Figures 1-2 are built directly from results/study.json and
results/nse_study_full.json. Figures 3-5 re-run a small, representative
training pass on one NSE snapshot (so they are reproducible from scratch).

Usage:
    python make_figures.py --fig 1      # synthetic fit-compliance frontier
    python make_figures.py --fig 2      # real-data 1359-day study overview
    python make_figures.py --fig 345    # fitted slice, hedge PnL, convergence
    python make_figures.py --fig all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # <repo>/research-paper/<dir>/figures
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO))

# --- publication house style -------------------------------------------------
plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.7,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Consistent, colourblind-safe palette: ArbNet (blue), Ackerer (vermilion), BS (green).
COL = {"arbnet": "#1f4e96", "ackerer": "#c8442a", "bs": "#3a8a4f"}
MK = {"arbnet": "o", "ackerer": "s", "bs": "D"}
# matplotlib mathtext only -- no LaTeX -- so use a real en-dash, not "--".
LABEL = {"arbnet": "ArbNet", "ackerer": "Ackerer-Net", "bs": "Black–Scholes"}


def _save(fig, name):
    out = HERE / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


# =============================================================================
# Figure 1 -- fit-compliance frontier (synthetic + real)
# =============================================================================
def _frontier_stats(recs, models):
    """Per-model (mean, std) of the butterfly+calendar violation rate and RMSE."""
    out = {}
    for M, m in models:
        rs = [r for r in recs if r["model"] == M]
        rmse = np.array([r["price_rmse"] for r in rs], dtype=float)
        viol = np.array([100.0 * (r["butterfly_rate"] + r["calendar_rate"]) for r in rs])
        out[m] = (viol.mean(), viol.std(), rmse.mean(), rmse.std())
    return out


def fig1_pareto():
    syn = json.load(open(RESULTS / "study.json"))["records"]
    real = json.load(open(RESULTS / "nse_study_full.json"))["records"]
    st_syn = _frontier_stats(syn, [("ArbNet", "arbnet"), ("Ackerer", "ackerer"), ("BS", "bs")])
    st_real = _frontier_stats(real, [("arbnet", "arbnet"), ("ackerer", "ackerer"), ("bs", "bs")])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fig.subplots_adjust(wspace=0.30, bottom=0.26)
    handles = None
    for ax, st, ttl in [
        (axes[0], st_syn, "(a) Synthetic rough-Bergomi (30 trials)"),
        (axes[1], st_real, "(b) Real NSE walk-forward (1359 days)"),
    ]:
        for m in ("arbnet", "ackerer", "bs"):
            vx, vsd, ry, rsd = st[m]
            xerr = np.array([[min(vsd, vx)], [vsd]])      # clip lower whisker at 0
            ax.errorbar(vx, ry, xerr=xerr, yerr=rsd, fmt=MK[m], ms=8,
                        color=COL[m], ecolor=COL[m], elinewidth=1.0, capsize=2.5,
                        mfc=COL[m], mec="white", mew=0.7, label=LABEL[m], zorder=3)
        ax.axvline(0.0, ls=(0, (2, 2)), lw=0.8, color="0.55", zorder=1)
        ax.set_xlabel("mean butterfly + calendar\nviolation rate (% of grid points)")
        ax.set_title(ttl)
        ax.set_xlim(left=-max(0.06, 0.05 * ax.get_xlim()[1]))
        # concise label by the ArbNet marker (sits in empty space to its right)
        ya = st["arbnet"][2]
        ax.annotate("ArbNet: 0% violations\n(guaranteed, every trial)",
                    xy=(0, ya), xytext=(0.30, 0.62), textcoords="axes fraction",
                    fontsize=7.2, color=COL["arbnet"], va="center")
        handles = ax.get_legend_handles_labels()
    axes[0].set_ylabel("price RMSE (INR)")
    fig.legend(*handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.3, columnspacing=1.8)
    fig.suptitle("Fit–compliance frontier: exact arbitrage-freeness versus price fit",
                 y=1.0, fontsize=10)
    _save(fig, "fig1_pareto.pdf")


# =============================================================================
# Figure 2 -- real-data 1359-day study overview
# =============================================================================
def fig2_real_study():
    import pandas as pd
    d = json.load(open(RESULTS / "nse_study_full.json"))
    df = pd.DataFrame(d["records"])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.45, wspace=0.32)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_yr = fig.add_subplot(gs[1, 0])
    ax_cmp = fig.add_subplot(gs[1, 1])

    # (a) per-day price RMSE, 21-day rolling mean, 2019-2024
    for m in ("arbnet", "ackerer", "bs"):
        sub = df[df.model == m].sort_values("date")
        roll = sub["price_rmse"].rolling(21, min_periods=5).mean()
        ax_ts.plot(sub["date"], roll, color=COL[m], label=LABEL[m], lw=1.3)
    ax_ts.set_ylabel("price RMSE (INR)\n21-day rolling mean")
    ax_ts.set_title("(a) Daily fit error across the 1359-day NSE walk-forward (2019–2024)")
    ax_ts.legend(loc="upper left", ncol=3, handletextpad=0.3, columnspacing=1.2)
    ax_ts.margins(x=0.01)

    # (b) RMSE by calendar year
    years = sorted(df["year"].unique())
    w = 0.26
    for i, m in enumerate(("arbnet", "ackerer", "bs")):
        means = [df[(df.model == m) & (df.year == y)]["price_rmse"].mean() for y in years]
        ax_yr.bar(np.arange(len(years)) + (i - 1) * w, means, w,
                  color=COL[m], label=LABEL[m], edgecolor="white")
    ax_yr.set_xticks(np.arange(len(years)))
    ax_yr.set_xticklabels([str(y) for y in years])
    ax_yr.set_ylabel("mean price RMSE (INR)")
    ax_yr.set_title("(b) Fit error by year")

    # (c) static-arbitrage compliance: days with a butterfly/calendar violation
    viol = {}
    for m in ("arbnet", "ackerer", "bs"):
        sub = df[df.model == m]
        viol[m] = int(((sub["butterfly_rate"] > 0) | (sub["calendar_rate"] > 0)).sum())
    bars = ax_cmp.bar([LABEL[m] for m in ("arbnet", "ackerer", "bs")],
                      [viol[m] for m in ("arbnet", "ackerer", "bs")],
                      color=[COL[m] for m in ("arbnet", "ackerer", "bs")],
                      edgecolor="white", width=0.6)
    for b, m in zip(bars, ("arbnet", "ackerer", "bs")):
        ax_cmp.text(b.get_x() + b.get_width() / 2, b.get_height() + 18,
                    f"{viol[m]}", ha="center", va="bottom", fontsize=8.5)
    ax_cmp.set_ylabel("trading days with a\nbutterfly/calendar violation")
    ax_cmp.set_ylim(0, 1359)
    ax_cmp.set_title("(c) Static-arbitrage compliance (of 1359 days)")
    ax_cmp.tick_params(axis="x", labelsize=7.5)

    _save(fig, "fig2_real_study.pdf")


# =============================================================================
# Figures 3-5 -- one representative NSE snapshot, three models trained once
# =============================================================================
def _train_representative(date="2023-06-01", n_epochs=150):
    """Train ArbNet / Ackerer / BS on one real NSE snapshot; return everything
    figures 3-5 need (models, features, snapshot, per-epoch loss histories)."""
    import torch
    from arbnet.data import load_snapshot, apply_quality_filters, FilterConfig, build_features
    from arbnet.data.iv import implied_vol_newton
    from arbnet.models import ArbNet, AckererSoftPenaltyNet, BlackScholesPricer
    from arbnet.utils import default_arbnet_config, RunConfig, set_seed
    from arbnet.train import train_pricer
    import pandas as pd

    set_seed(0)
    snap = load_snapshot(pd.Timestamp(date)).call_subset()
    filt, _ = apply_quality_filters(snap, FilterConfig())
    feats = build_features(filt)
    feats["iv"] = implied_vol_newton(feats["price"], feats["S"], feats["K"],
                                     feats["T"], feats["r"], feats.get("q"))
    out = {"snap": filt, "feats": feats, "models": {}, "loss": {}}

    arb = ArbNet(default_arbnet_config())
    diag = train_pricer(arb, dict(feats), RunConfig(seed=0, n_epochs=n_epochs,
                        batch_size=64, lr=1e-2, lambda_iv=0.0), verbose=False)
    out["models"]["arbnet"], out["loss"]["arbnet"] = arb, diag["loss_history"]

    import torch as _t
    ack = AckererSoftPenaltyNet()
    K_pen = float(filt.spot) * _t.exp(_t.linspace(-0.3, 0.3, 21, dtype=_t.float32))
    T_pen = _t.tensor(sorted({round(float(t), 6) for t in feats["T"].tolist()}),
                      dtype=_t.float32)
    diag = train_pricer(ack, dict(feats), RunConfig(seed=0, n_epochs=n_epochs,
                        batch_size=64, lr=1e-2, lambda_iv=0.0, lambda_arb=1.0),
                        soft_penalty_grid={"K_grid": K_pen, "T_grid": T_pen}, verbose=False)
    out["models"]["ackerer"], out["loss"]["ackerer"] = ack, diag["loss_history"]

    bs = BlackScholesPricer()
    diag = train_pricer(bs, dict(feats), RunConfig(seed=0, n_epochs=min(n_epochs, 80),
                        batch_size=64, lr=2e-2, lambda_iv=0.0), verbose=False)
    out["models"]["bs"], out["loss"]["bs"] = bs, diag["loss_history"]
    return out


def fig3_fitted_slice(ctx):
    import torch
    from arbnet.data.iv import implied_vol_newton
    snap, models = ctx["snap"], ctx["models"]
    S, r, q = float(snap.spot), float(snap.risk_free_rate), float(snap.dividend_yield)

    # pick the maturity with the most quotes in the 20-45 DTE band
    T = np.asarray(snap.times_to_expiry, float)
    K = np.asarray(snap.strikes, float)
    P = np.asarray(snap.prices, float)
    band = (T >= 20 / 365) & (T <= 45 / 365)
    Tu, cnt = np.unique(np.round(T[band], 6), return_counts=True)
    T0 = float(Tu[cnt.argmax()])
    sl = np.isclose(T, T0, atol=1e-6) & (P > 0.05)
    order = np.argsort(K[sl])
    Km, Pm = K[sl][order], P[sl][order]
    mny = Km / S

    Kg = np.linspace(Km.min(), Km.max(), 200)
    tt = torch.tensor(Kg, dtype=torch.float32)
    Sb = torch.full_like(tt, S); rb = torch.full_like(tt, r); qb = torch.full_like(tt, q)
    Tb = torch.full_like(tt, T0)
    curves = {}
    for m, model in models.items():
        with torch.no_grad():
            c = model(tt, Tb, Sb, rb, qb)["price"].numpy()
        iv = implied_vol_newton(torch.tensor(c), Sb, tt, Tb, rb, qb).numpy()
        rnd = np.gradient(np.gradient(c, Kg), Kg) * np.exp(r * T0)
        curves[m] = dict(price=c, iv=iv, rnd=rnd)

    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.2))
    fig.subplots_adjust(hspace=0.42, wspace=0.30)
    # (a) price
    ax[0, 0].scatter(mny, Pm, s=20, facecolor="0.25", edgecolor="white",
                     linewidth=0.4, zorder=5, label="NSE quotes")
    for m in ("arbnet", "ackerer", "bs"):
        ax[0, 0].plot(Kg / S, curves[m]["price"], color=COL[m], label=LABEL[m])
    ax[0, 0].set_xlabel("moneyness $K/S$"); ax[0, 0].set_ylabel("call price (INR)")
    ax[0, 0].set_title("(a) Fitted call prices"); ax[0, 0].legend(handletextpad=0.3)
    # (b) residuals -- model minus market, interpolated to quote strikes.
    # Markers only: the point-to-point jaggedness is genuine quote noise, and a
    # connecting line would render it as visual clutter rather than signal.
    for m in ("arbnet", "ackerer", "bs"):
        resid = np.interp(Km, Kg, curves[m]["price"]) - Pm
        ax[0, 1].plot(mny, resid, color=COL[m], marker=MK[m], ms=3.2, lw=0,
                      mfc=COL[m], mec="white", mew=0.3, alpha=0.85, label=LABEL[m])
    ax[0, 1].axhline(0, color="0.4", lw=0.7)
    ax[0, 1].set_xlabel("moneyness $K/S$"); ax[0, 1].set_ylabel("price residual (INR)")
    ax[0, 1].set_title("(b) Pricing residuals")
    # (c) implied-vol smile
    ivm = implied_vol_newton(torch.tensor(Pm, dtype=torch.float32),
                             torch.full((len(Pm),), S), torch.tensor(Km, dtype=torch.float32),
                             torch.full((len(Pm),), T0), torch.full((len(Pm),), r),
                             torch.full((len(Pm),), q)).numpy()
    win = (Kg / S >= 0.85) & (Kg / S <= 1.15)
    wm = (mny >= 0.85) & (mny <= 1.15)
    ax[1, 0].scatter(mny[wm], ivm[wm], s=20, facecolor="0.25", edgecolor="white",
                     linewidth=0.4, zorder=5, label="NSE quotes")
    for m in ("arbnet", "ackerer", "bs"):
        ax[1, 0].plot(Kg[win] / S, curves[m]["iv"][win], color=COL[m], label=LABEL[m])
    ax[1, 0].set_xlabel("moneyness $K/S$"); ax[1, 0].set_ylabel(r"implied volatility")
    ax[1, 0].set_title("(c) Implied-volatility smile")
    # (d) risk-neutral density. ArbNet keeps the intrinsic kink at the forward,
    # so its numerically-differentiated RND carries a Dirac atom there (Cor. 4.2);
    # we clip the axis so the absolutely-continuous parts stay legible.
    F0 = S * np.exp((r - q) * T0)
    smooth_max = max(float(np.nanmax(np.clip(curves[m]["rnd"][win], 0, None)))
                     for m in ("ackerer", "bs"))
    for m in ("arbnet", "ackerer", "bs"):
        ax[1, 1].plot(Kg[win] / S, np.clip(curves[m]["rnd"][win], 0, None),
                      color=COL[m], label=LABEL[m])
    ax[1, 1].set_ylim(0, smooth_max * 1.7)
    ax[1, 1].annotate("ArbNet: point mass\nat the forward",
                      xy=(F0 / S, smooth_max * 1.62),
                      xytext=(F0 / S + 0.045, smooth_max * 1.15),
                      fontsize=6.8, color=COL["arbnet"], va="center",
                      arrowprops=dict(arrowstyle="->", color=COL["arbnet"], lw=0.8))
    ax[1, 1].set_xlabel("moneyness $K/S$"); ax[1, 1].set_ylabel("risk-neutral density")
    ax[1, 1].set_title("(d) Implied risk-neutral density")
    fig.suptitle(f"Representative fit on a real NSE snapshot "
                 f"(T ≈ {T0*365:.0f} days, spot {S:,.0f})", y=1.005, fontsize=9.5)
    _save(fig, "fig3_fitted_slice.pdf")


def fig4_hedge_pnl(ctx):
    import torch
    from arbnet.models import RoughBergomiSimulator
    from arbnet.hedging import hedge_pnl_delta
    snap, models = ctx["snap"], ctx["models"]
    S, r, q = float(snap.spot), float(snap.risk_free_rate), float(snap.dividend_yield)
    T_h = 30 / 365.0
    sim = RoughBergomiSimulator(H=0.10, eta=1.5, rho=-0.7, xi0=0.04, dtype=torch.float64)
    t_grid = torch.linspace(0.0, T_h, 16, dtype=torch.float64)
    S_paths, _ = sim.simulate(2000, t_grid, S0=S, r=r, q=q, seed=0)
    atm = (torch.tensor([S]), torch.tensor([T_h]), torch.tensor([S]),
           torch.tensor([r]), torch.tensor([q]))

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for m in ("arbnet", "ackerer", "bs"):
        model = models[m]
        with torch.no_grad():
            if hasattr(model, "implied_vol"):
                iv = model.implied_vol(*atm)
                sig = float(iv.item()) if not torch.isnan(iv).any() else 0.2
            else:
                o = model(*atm)
                sig = float(o["iv"].item()) if "iv" in o else 0.2
        pnl = hedge_pnl_delta(S_paths.float(), torch.tensor(S), torch.tensor(T_h),
                              torch.tensor(r), torch.tensor(sig),
                              dt=float(T_h / 15), cost_bps=2.0).numpy()
        # CVaR_95 of the loss = mean of the worst (largest-loss) 5% of paths.
        losses = -pnl
        k = max(1, int(0.05 * len(losses)))
        cvar95 = float(np.mean(np.sort(losses)[-k:]))
        ax.hist(pnl, bins=60, density=True, histtype="stepfilled",
                color=COL[m], alpha=0.30, edgecolor="none")
        ax.hist(pnl, bins=60, density=True, histtype="step", color=COL[m], lw=1.3,
                label=f"{LABEL[m]} (CVaR$_{{95}}$ = {cvar95:,.0f} INR)")
        ax.axvline(-cvar95, color=COL[m], ls=(0, (3, 2)), lw=1.0, zorder=1)
    ax.axvline(0, color="0.4", lw=0.7)
    ax.set_xlabel("terminal delta-hedged PnL (INR)")
    ax.set_ylabel("density")
    ax.set_title("Delta-hedge PnL on 2000 rough-Bergomi paths (30-day horizon)")
    leg = ax.legend(loc="upper left", handletextpad=0.4, frameon=True,
                    framealpha=0.95, edgecolor="none")
    leg.get_frame().set_facecolor("white")
    _save(fig, "fig4_hedge_pnl.pdf")


def fig5_convergence(ctx):
    loss = ctx["loss"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.subplots_adjust(wspace=0.30)
    for m in ("arbnet", "ackerer", "bs"):
        h = np.asarray(loss[m], dtype=float)
        ep = np.arange(1, len(h) + 1)
        ax[0].plot(ep, h, color=COL[m], label=LABEL[m])
        ax[1].plot(ep, h / h[0], color=COL[m], label=LABEL[m])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("mean batch loss")
    ax[0].set_title("(a) Training loss (log scale)")
    ax[0].legend(handletextpad=0.3)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("loss / initial loss")
    ax[1].set_title("(b) Loss normalised by epoch-1 value")
    _save(fig, "fig5_convergence.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fig", default="all", help="1 | 2 | 345 | all")
    p.add_argument("--date", default="2023-06-01", help="NSE snapshot for figs 3-5")
    p.add_argument("--epochs", type=int, default=150)
    args = p.parse_args()

    if args.fig in ("1", "all"):
        print("figure 1 (synthetic fit-compliance frontier)")
        fig1_pareto()
    if args.fig in ("2", "all"):
        print("figure 2 (real-data 1359-day study)")
        fig2_real_study()
    if args.fig in ("345", "all"):
        print(f"figures 3-5 (representative fit on NSE {args.date})")
        ctx = _train_representative(args.date, args.epochs)
        fig3_fitted_slice(ctx)
        fig4_hedge_pnl(ctx)
        fig5_convergence(ctx)
    print("done.")


if __name__ == "__main__":
    main()
