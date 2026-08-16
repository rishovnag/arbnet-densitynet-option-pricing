#!/usr/bin/env python3
"""Regenerate the result figures of the paper from committed artifacts.

Covers Figures 3, 4, 6, 7, 8 and 9. Figure 1 is TikZ inside the manuscript;
Figure 2 (the butterfly-second-derivative box) and Figure 5 (the penalty-weight
sweep) are produced by the earlier diagnostic scripts.

Inputs (all under results/):
    nse_study_v4_skew.json          real study, context run      -> Figs 3, 4
    study_v4.json                   synthetic study, 30 trials   -> Fig 4a
    vm_sweep.json                   dispersion sweep, 68 days    -> Fig 7d
    v4_density_params.csv           per-day fitted density params-> Fig 7c
plus the committed NSE data for the single-day panels (Figs 6, 7a-b, 8, 9),
which refit the five models on one snapshot -- context-free, seed 0, the same
optimiser and schedule as the study.

    python scripts/make_figures.py --outdir research-paper/.../figures

Everything is deterministic given the committed data and the pinned seed.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arbnet.data import load_snapshot, apply_quality_filters, FilterConfig, build_features
from arbnet.data.iv import implied_vol_newton
from arbnet.models import (ArbNet, AckererSoftPenaltyNet, BlackScholesPricer,
                           DensityNet, DensityNetConfig, RoughBergomiSimulator)
from arbnet.utils import default_arbnet_config, RunConfig, set_seed
from arbnet.train import train_pricer
from arbnet.hedging import hedge_pnl_delta

SNAP_DATE = "2023-06-01"
C = {"skew": "#1a7a4a", "fix": "#7bb661", "arb": "#1f6fb4", "ack": "#c0392b", "bs": "#7f7f7f",
     "ackm": "#d98a6a", "bump": "#8e5fa8"}
LBL = {"skew": "DensityNet", "fix": r"DensityNet$_{h\equiv1}$", "arb": "ArbNet",
       "ack": "Ackerer-Net", "bs": "Black--Scholes",
       "ackm": "Ackerer (matched)", "bump": "ArbNet + bump"}
LS = {"skew": "-", "fix": "-", "arb": "-", "ack": "--", "bs": ":"}
ORD = ["arb", "skew", "fix", "ack", "bs"]
SERIF = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"}
BIG = dict(SERIF, **{"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5})
SMALL = {"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.grid": False}
TEXTW = 443.273 / 72.0


def fit_snapshot(date=SNAP_DATE):
    """Refit the five models, context-free, on one NSE snapshot."""
    snap = load_snapshot(pd.Timestamp(date)).call_subset()
    filt, _ = apply_quality_filters(snap, FilterConfig())
    feats = build_features(filt)
    feats["iv"] = implied_vol_newton(feats["price"], feats["S"], feats["K"],
                                     feats["T"], feats["r"], feats.get("q"))
    S0, r0, q0 = float(snap.spot), float(snap.risk_free_rate), float(snap.dividend_yield)
    models, hist = {}, {}
    for k in ORD:
        set_seed(0)
        grid = None
        if k == "skew":
            m, cfg = DensityNet(DensityNetConfig(0, 8, skew_clock=True)), RunConfig(seed=0, n_epochs=150, batch_size=64, lr=1e-2, lambda_iv=0.0)
        elif k == "fix":
            m, cfg = DensityNet(DensityNetConfig(0, 8)), RunConfig(seed=0, n_epochs=150, batch_size=64, lr=1e-2, lambda_iv=0.0)
        elif k == "arb":
            m, cfg = ArbNet(default_arbnet_config(context_dim=0)), RunConfig(seed=0, n_epochs=150, batch_size=64, lr=1e-2, lambda_iv=0.0)
        elif k == "ack":
            m = AckererSoftPenaltyNet()
            cfg = RunConfig(seed=0, n_epochs=150, batch_size=64, lr=1e-2, lambda_iv=0.0, lambda_arb=1.0)
            grid = {"K_grid": S0 * torch.exp(torch.linspace(-0.3, 0.3, 21)),
                    "T_grid": torch.tensor(sorted({round(float(t), 6) for t in feats["T"].tolist()}))}
        else:
            m, cfg = BlackScholesPricer(), RunConfig(seed=0, n_epochs=80, batch_size=64, lr=2e-2, lambda_iv=0.0)
        d = train_pricer(m, feats, cfg, soft_penalty_grid=grid, verbose=False)
        models[k], hist[k] = m, d["loss_history"]
    return snap, feats, models, hist, S0, r0, q0


def _iv_curve(model, K, T, S0, r0, q0, jump=0.05):
    n = K.shape[0]
    args = (torch.full((n,), T), torch.full((n,), S0), torch.full((n,), r0), torch.full((n,), q0))
    if hasattr(model, "implied_vol"):
        iv = model.implied_vol(K, *args, None).numpy()
    else:
        with torch.no_grad():
            p = model(K, *args, None)["price"].numpy()
        iv = implied_vol_newton(torch.tensor(p), args[1], K, args[0], args[2], args[3]).numpy()
    iv = np.asarray(iv, float)
    bad = ~np.isfinite(iv) | (iv <= 1e-3) | (iv >= 1.5) | (np.abs(np.diff(iv, prepend=iv[0])) > jump)
    return np.where(bad, np.nan, iv)


def fig3(out, real):
    plt.rcParams.update(SMALL)
    recs = real["records"]
    by = {}
    for r in recs:
        by.setdefault(r["date"], {})[r["model"]] = r["price_rmse"]
    dates = sorted(by)
    key = {"skew": "arbnet_density_skew", "fix": "arbnet_density", "arb": "arbnet", "ack": "ackerer", "bs": "bs"}
    ser = {k: np.array([by[d][v] for d in dates]) for k, v in key.items()}
    years = np.array([d[:4] for d in dates])
    fig, ax = plt.subplots(1, 3, figsize=(TEXTW, 2.05))
    xr = np.arange(len(dates))[20:]
    for k in ["skew", "fix", "arb", "ack", "bs"]:
        ax[0].plot(xr, np.convolve(ser[k], np.ones(21) / 21, mode="valid"), color=C[k], lw=1.0, label=LBL[k])
    tks, seen = [], set()
    for i, d in enumerate(dates):
        if d[5:7] == "01" and d[8:10] <= "05" and d[:4] not in seen:
            seen.add(d[:4]); tks.append(i)
    ax[0].set_xticks(tks); ax[0].set_xticklabels([dates[i][:4] for i in tks])
    ax[0].set_ylabel("price RMSE (INR)"); ax[0].set_title("(a) 21-day rolling held-out RMSE", pad=4)
    ax[0].legend(frameon=False, loc="upper left")
    ys = sorted(set(years)); w = 0.16
    for j, k in enumerate(["skew", "fix", "arb", "ack", "bs"]):
        ax[1].bar(np.arange(len(ys)) + (j - 2) * w, [ser[k][years == y].mean() for y in ys],
                  width=w, color=C[k])
    ax[1].set_xticks(np.arange(len(ys))); ax[1].set_xticklabels(ys)
    ax[1].set_ylabel("mean held-out RMSE (INR)"); ax[1].set_title("(b) By calendar year", pad=4)
    agg = real["aggregate"]
    viol = [agg[key[k]]["guaranteed_violation_days"] for k in ["skew", "fix", "arb", "ack", "bs"]]
    ax[2].bar(np.arange(5), viol, color=[C[k] for k in ["skew", "fix", "arb", "ack", "bs"]])
    ax[2].set_xticks(np.arange(5))
    ax[2].set_xticklabels(["Dens.", r"Dens.$_{h\equiv1}$", "ArbNet", "Ackerer", "BS"], rotation=30, ha="right")
    ax[2].set_ylabel("days with a violation"); ax[2].set_ylim(0, 1400)
    ax[2].set_title("(c) Violation days (of 1,359)", pad=4)
    for i, v in enumerate(viol):
        ax[2].text(i, v + 30, str(v), ha="center", fontsize=6.5)
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.4); fig.savefig(f"{out}/Fig3.pdf"); plt.close(fig)


def fig4(out, real, syn):
    plt.rcParams.update(SMALL)
    sm = {"skew": "ArbNet_density_skew", "fix": "ArbNet_density", "arb": "ArbNet", "ack": "Ackerer",
          "bs": "BS", "ackm": "Ackerer_matched", "bump": "ArbNet_bump"}
    rm = {"skew": "arbnet_density_skew", "fix": "arbnet_density", "arb": "arbnet", "ack": "ackerer",
          "bs": "bs", "ackm": "ackerer_matched", "bump": "arbnet_bump"}
    mk = {"skew": "o", "fix": "o", "arb": "o", "ack": "^", "bs": "s", "ackm": "^", "bump": "v"}
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.3))
    for k in ["skew", "fix", "arb", "ack", "ackm", "bump", "bs"]:
        a0, a1 = syn["aggregate"][sm[k]], real["aggregate"][rm[k]]
        ax[0].scatter(a0["butterfly_rate_mean"] * 100, a0["price_rmse_mean"], s=32, color=C[k],
                      marker=mk[k], zorder=3, label=LBL[k], edgecolor="white", linewidth=0.4)
        ax[1].scatter(a1["guaranteed_butterfly_rate_mean"] * 100, a1["price_rmse_mean"], s=32,
                      color=C[k], marker=mk[k], zorder=3, edgecolor="white", linewidth=0.4)
    for a, ttl, yl in [(ax[0], "(a) Synthetic, 30 trials", "price RMSE (INR)"),
                       (ax[1], "(b) Real NSE, 1,359 days", "held-out price RMSE (INR)")]:
        a.set_xscale("symlog", linthresh=0.05); a.axvline(0, color="#999", lw=0.8, ls=":")
        a.set_xlabel("mean butterfly violation rate (\\% of grid points)")
        a.set_ylabel(yl); a.set_title(ttl, pad=4); a.set_xlim(-0.012, 40)
        a.set_xticks([0, 0.05, 0.5, 5]); a.set_xticklabels(["0", "0.05", "0.5", "5"])
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    ax[0].legend(frameon=False, loc="upper left", handletextpad=0.4, borderpad=0.2)
    fig.tight_layout(pad=0.4); fig.savefig(f"{out}/Fig4.pdf"); plt.close(fig)


def fig6(out, feats, models, S0, r0, q0):
    plt.rcParams.update(BIG)
    Ts = np.array(sorted({round(float(t), 8) for t in feats["T"].tolist()}))
    T = float(Ts[np.argmin(np.abs(Ts - 28 / 365.0))])
    sel = np.abs(feats["T"].numpy() - T) < 1e-4
    K = feats["K"].numpy()[sel]; P = feats["price"].numpy()[sel]
    o = np.argsort(K); K, P = K[o], P[o]; mn = K / S0
    FT = S0 * math.exp((r0 - q0) * T)
    fig, AX = plt.subplots(2, 2, figsize=(9.2, 6.4))
    Kg = torch.linspace(float(K.min()), float(K.max()), 400); n = Kg.shape[0]
    A = (torch.full((n,), T), torch.full((n,), S0), torch.full((n,), r0), torch.full((n,), q0), None)
    ax = AX[0, 0]
    for k in ORD:
        with torch.no_grad():
            ax.plot(Kg.numpy() / S0, models[k](Kg, *A)["price"].numpy(), color=C[k], ls=LS[k], lw=1.4, label=LBL[k])
    ax.scatter(mn, P, s=11, color="#333", zorder=5, label="NSE quotes")
    ax.set_xlabel("moneyness $K/S$"); ax.set_ylabel("call price (INR)"); ax.set_title("(a) Fitted call prices")
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")
    ax = AX[0, 1]
    for k in ORD:
        with torch.no_grad():
            pq = models[k](torch.tensor(K, dtype=torch.float32), torch.full((len(K),), T),
                           torch.full((len(K),), S0), torch.full((len(K),), r0),
                           torch.full((len(K),), q0), None)["price"].numpy()
        ax.scatter(mn, pq - P, s=9, color=C[k], alpha=0.85)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-150, 150)
    ax.set_xlabel("moneyness $K/S$"); ax.set_ylabel("price residual (INR)"); ax.set_title("(b) Pricing residuals")
    ax = AX[1, 0]
    ivq = implied_vol_newton(torch.tensor(P), torch.full((len(K),), S0), torch.tensor(K),
                             torch.full((len(K),), T), torch.full((len(K),), r0),
                             torch.full((len(K),), q0)).numpy()
    Kiv = torch.linspace(0.85 * S0, 1.13 * S0, 320)
    for k in ORD:
        ax.plot(Kiv.numpy() / S0, _iv_curve(models[k], Kiv, T, S0, r0, q0), color=C[k], ls=LS[k], lw=1.4, label=LBL[k])
    ax.scatter(mn, ivq, s=11, color="#333", zorder=5, label="NSE quotes")
    ax.set_xlim(0.63, 1.27); ax.set_ylim(0.0, 0.55)
    ax.set_xlabel("moneyness $K/S$"); ax.set_ylabel("implied volatility"); ax.set_title("(c) Implied-volatility smile")
    ax = AX[1, 1]
    Kd = torch.linspace(0.85 * S0, 1.15 * S0, 900, dtype=torch.float64); nd = Kd.shape[0]
    Ad = tuple(torch.full((nd,), v, dtype=torch.float64) for v in (T, S0, r0, q0)) + (None,)
    dK = float(Kd[1] - Kd[0])
    for k in ORD:
        m = models[k]; m.double()
        try:
            with torch.no_grad():
                p = m(Kd, *Ad)["price"].numpy()
        finally:
            m.float()
        rho = np.gradient(np.gradient(p, dK), dK) * math.exp(r0 * T)
        if k == "arb":
            ax.axvline(FT / S0, color=C[k], lw=2.2)
            ax.annotate("ArbNet: point mass at $K=F_T$", xy=(FT / S0, 0.00097),
                        xytext=(FT / S0 + 0.006, 0.00100), color=C[k], fontsize=8)
            msk = np.abs(Kd.numpy() - FT) > 2.5 * dK
            ax.plot(Kd.numpy()[msk] / S0, np.where(rho[msk] < 0, 0, rho[msk]), color=C[k], ls=LS[k], lw=1.2)
        else:
            ax.plot(Kd.numpy()[3:-3] / S0, rho[3:-3], color=C[k], ls=LS[k], lw=1.4)
    ax.set_xlim(0.85, 1.15); ax.set_ylim(0, 0.00115)
    ax.set_xlabel("moneyness $K/S$"); ax.set_ylabel("risk-neutral density")
    ax.set_title("(d) Implied risk-neutral density")
    fig.tight_layout(pad=0.6); fig.savefig(f"{out}/Fig6.pdf"); plt.close(fig)


def fig7(out, feats, models, S0, r0, q0, sweep, params_csv):
    plt.rcParams.update(BIG)
    Ts = np.array(sorted({round(float(t), 8) for t in feats["T"].tolist()})); Tsh = float(Ts[0])
    fig, AX = plt.subplots(2, 2, figsize=(9.6, 7.0))
    ax = AX[0, 0]
    sel = np.abs(feats["T"].numpy() - Tsh) < 1e-6
    K = feats["K"].numpy()[sel]; P = feats["price"].numpy()[sel]
    o = np.argsort(K); K, P = K[o], P[o]
    ivq = implied_vol_newton(torch.tensor(P), torch.full((len(K),), S0), torch.tensor(K),
                             torch.full((len(K),), Tsh), torch.full((len(K),), r0),
                             torch.full((len(K),), q0)).numpy()
    ax.scatter(K / S0, ivq, s=13, color="#333", zorder=5, label="NSE quotes")
    Kg = torch.linspace(0.93 * S0, 1.07 * S0, 300)
    for k in ("fix", "skew"):
        ax.plot(Kg.numpy() / S0, _iv_curve(models[k], Kg, Tsh, S0, r0, q0, jump=0.03),
                color=C[k], lw=1.6, label=LBL[k])
    ax.set_xlim(0.93, 1.07); ax.set_ylim(0, 1.05)
    ax.set_xlabel("moneyness $K/S$"); ax.set_ylabel("implied volatility")
    ax.set_title("(a) Smile at the shortest listed maturity (7 days)")
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")
    ax = AX[0, 1]
    Tg = np.geomspace(1 / 365.0, 0.30, 90); nn = len(Tg)
    for k in ("fix", "skew"):
        iv = models[k].implied_vol(torch.full((nn,), S0), torch.tensor(Tg, dtype=torch.float32),
                                   torch.full((nn,), S0), torch.full((nn,), r0),
                                   torch.full((nn,), q0), None).numpy()
        ax.plot(Tg * 365.0, np.where(np.isfinite(iv) & (iv > 1e-3), iv, np.nan), color=C[k], lw=1.6, label=LBL[k])
    mkt = []
    for T in Ts:
        s = np.abs(feats["T"].numpy() - T) < 1e-6
        Kk, Pp = feats["K"].numpy()[s], feats["price"].numpy()[s]
        j = int(np.argmin(np.abs(Kk - S0)))
        mkt.append(implied_vol_newton(torch.tensor([Pp[j]]), torch.tensor([S0]), torch.tensor([Kk[j]]),
                                      torch.tensor([float(T)]), torch.tensor([r0]), torch.tensor([q0])).item())
    ax.scatter(Ts * 365.0, mkt, s=28, color="#333", zorder=5, marker="D", label="NSE at-the-money")
    V = float(models["fix"].mixture_diagnostics(None)["V_m"])
    ax.plot(Tg * 365.0, np.sqrt(max(V, 1e-12) / Tg), color=C["fix"], ls=":", lw=1.3, label=r"floor $\sqrt{V_m/T}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("maturity (days)"); ax.set_ylabel("at-the-money implied volatility")
    ax.set_title("(b) At-the-money term structure"); ax.legend(frameon=True, framealpha=0.9, loc="upper right")
    ax = AX[1, 0]
    rows = list(csv.DictReader(open(params_csv)))
    bins = np.geomspace(1e-7, 1.0, 60)
    for run, mdl, col, ls, lab in [("ctx", "arbnet_density", C["fix"], "-", r"DensityNet$_{h\equiv1}$, context"),
                                   ("noctx", "arbnet_density", C["fix"], ":", r"DensityNet$_{h\equiv1}$, context-free"),
                                   ("ctx", "arbnet_density_skew", C["skew"], "-", "DensityNet, context"),
                                   ("noctx", "arbnet_density_skew", C["skew"], ":", "DensityNet, context-free")]:
        v = np.array([float(r["V_m"]) for r in rows if r["run"] == run and r["model"] == mdl])
        ax.hist(v, bins=bins, histtype="step", lw=1.6, color=col, ls=ls, label=lab)
    ax.set_xscale("log"); ax.set_ylabel("trading days")
    ax.set_xlabel(r"fitted mixing dispersion $V_m=\mathrm{Var}(\log \bar m_\theta)$")
    ax.set_title("(c) Fitted skew budget, 1,359 days"); ax.legend(frameon=True, framealpha=0.9, loc="upper left")
    ax = AX[1, 1]
    sw = sweep["aggregate"]
    order = ["fixed_0", "fixed_0.0001", "fixed_0.001", "fixed_0.004", "fixed_0.01", "fixed_0.03"]
    labs = ["0", r"$10^{-4}$", r"$10^{-3}$", r"$4\!\times\!10^{-3}$", r"$10^{-2}$", r"$3\!\times\!10^{-2}$"]
    xs = [sw[k]["iv_long"] for k in order]; ys = [sw[k]["iv_short"] for k in order]
    ax.plot(xs, ys, "-o", color=C["fix"], lw=1.5, ms=5, label=r"DensityNet$_{h\equiv1}$, $V_m$ pinned")
    for lab, x, y in zip(labs, xs, ys):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7, color="#3f6b32")
    ax.scatter([sw["fixed_free"]["iv_long"]], [sw["fixed_free"]["iv_short"]], s=55, marker="s",
               color=C["fix"], edgecolor="k", linewidth=0.8, zorder=6, label=r"DensityNet$_{h\equiv1}$, $V_m$ free")
    ax.scatter([sw["skew"]["iv_long"]], [sw["skew"]["iv_short"]], s=110, marker="*",
               color=C["skew"], edgecolor="k", linewidth=0.8, zorder=6, label="DensityNet (fitted skew clock)")
    ax.set_xlabel("held-out IV RMSE, maturities $>21$ days")
    ax.set_ylabel(r"held-out IV RMSE, maturities $\leq 21$ days")
    ax.set_title("(d) The dispersion frontier, 68-day subsample")
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")
    fig.tight_layout(pad=0.7); fig.savefig(f"{out}/Fig7.pdf"); plt.close(fig)


def fig8(out, models, S0, r0, q0):
    plt.rcParams.update(BIG)
    sim = RoughBergomiSimulator(H=0.10, eta=1.5, rho=-0.7, xi0=0.04, dtype=torch.float64)
    Th = 30 / 365.0
    Sp, _ = sim.simulate(2000, torch.linspace(0.0, Th, 16, dtype=torch.float64),
                         S0=S0, r=r0, q=q0, seed=0)
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for k in ORD:
        with torch.no_grad():
            iv = models[k].implied_vol(torch.tensor([S0]), torch.tensor([Th]), torch.tensor([S0]),
                                       torch.tensor([r0]), torch.tensor([q0]), None) \
                if hasattr(models[k], "implied_vol") else None
        sig = float(iv.item()) if (iv is not None and torch.isfinite(iv).all()) else 0.20
        pnl = hedge_pnl_delta(Sp.float(), torch.tensor(S0), torch.tensor(Th), torch.tensor(r0),
                              torch.tensor(sig), dt=float(Th / 15), cost_bps=2.0).numpy()
        cv = -float(np.mean(np.sort(pnl)[:max(1, int(0.05 * len(pnl)))]))
        ax.hist(pnl, bins=90, histtype="step", color=C[k], lw=1.3, density=True,
                label=f"{LBL[k]} (CVaR$_{{95}}$ {cv:.0f})")
        ax.axvline(-cv, color=C[k], ls="--", lw=1.0)
    ax.set_xlabel("terminal delta-hedged PnL (INR)"); ax.set_ylabel("density")
    ax.legend(frameon=True, framealpha=0.9, loc="upper left")
    fig.tight_layout(pad=0.5); fig.savefig(f"{out}/Fig8.pdf"); plt.close(fig)


def fig9(out, hist):
    plt.rcParams.update(BIG)
    fig, AX = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for k in ORD:
        v = np.array(hist[k]); e = np.arange(1, len(v) + 1)
        AX[0].semilogy(e, v, color=C[k], ls=LS[k], lw=1.4, label=LBL[k])
        AX[1].semilogy(e, v / v[0], color=C[k], ls=LS[k], lw=1.4)
    AX[0].set_title("(a) Training loss, log scale"); AX[0].set_ylabel("mean batch loss")
    AX[1].set_title("(b) Loss normalised by epoch-1 value"); AX[1].set_ylabel("loss / initial loss")
    for a in AX:
        a.set_xlabel("epoch"); a.set_xlim(0, 152)
    AX[0].legend(frameon=True, framealpha=0.9, loc="upper right")
    fig.tight_layout(pad=0.5); fig.savefig(f"{out}/Fig9.pdf"); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="figures")
    p.add_argument("--results", default="results")
    p.add_argument("--only", nargs="*", default=None, help="subset, e.g. --only 3 4")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    want = set(a.only) if a.only else {"3", "4", "6", "7", "8", "9"}
    real = json.load(open(f"{a.results}/nse_study_v4_skew.json"))
    syn = json.load(open(f"{a.results}/study_v4.json"))
    sweep = json.load(open(f"{a.results}/vm_sweep.json"))
    if want & {"3"}: fig3(a.outdir, real); print("Fig3")
    if want & {"4"}: fig4(a.outdir, real, syn); print("Fig4")
    if want & {"6", "7", "8", "9"}:
        snap, feats, models, hist, S0, r0, q0 = fit_snapshot()
        if "6" in want: fig6(a.outdir, feats, models, S0, r0, q0); print("Fig6")
        if "7" in want: fig7(a.outdir, feats, models, S0, r0, q0, sweep,
                             f"{a.results}/v4_density_params.csv"); print("Fig7")
        if "8" in want: fig8(a.outdir, models, S0, r0, q0); print("Fig8")
        if "9" in want: fig9(a.outdir, hist); print("Fig9")


if __name__ == "__main__":
    main()
