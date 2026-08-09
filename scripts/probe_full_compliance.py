#!/usr/bin/env python3
"""Probe: fit the study's models on a handful of regime-spanning NSE days and
run the FULL static-arbitrage compliance diagnostic (monotonicity, digital
bound, upper bound, K->0 boundary) that the existing reports omit.

Question it answers: does ArbNet's structural violation (convex forward time
value => eventually-increasing call, boundary excess at K=0) bite INSIDE the
traded +-30% band on real fitted surfaces, or only in the extrapolation
region?  DensityNet should pass everything (it prices against a genuine
probability measure); Ackerer is penalised-only; BS is exact.

Mirrors scripts/train_nse.py per-day training exactly (no context).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbnet.data import load_snapshot, apply_quality_filters, FilterConfig, build_features
from arbnet.data.iv import implied_vol_newton
from arbnet.models import (ArbNet, AckererSoftPenaltyNet, BlackScholesPricer,
                           DensityNet, DensityNetConfig)
from arbnet.utils import default_arbnet_config, RunConfig, set_seed
from arbnet.train import train_pricer
from arbnet.arbitrage.full_compliance import full_compliance_report

DATES = ["2019-04-01", "2020-03-19", "2020-06-10", "2021-07-01",
         "2022-06-17", "2023-06-01", "2024-04-03"]
MODELS = ["arbnet", "arbnet_density", "ackerer", "bs"]
N_EPOCHS = 150


def _fit(name, feats, seed):
    if name == "arbnet":
        model = ArbNet(default_arbnet_config(context_dim=0))
        cfg = RunConfig(seed=seed, n_epochs=N_EPOCHS, batch_size=64, lr=1e-2, lambda_iv=0.0)
        train_pricer(model, feats, cfg, verbose=False)
    elif name == "arbnet_density":
        model = DensityNet(DensityNetConfig(context_dim=0, n_components=8))
        cfg = RunConfig(seed=seed, n_epochs=N_EPOCHS, batch_size=64, lr=1e-2, lambda_iv=0.0)
        train_pricer(model, feats, cfg, verbose=False)
    elif name == "ackerer":
        model = AckererSoftPenaltyNet()
        cfg = RunConfig(seed=seed, n_epochs=N_EPOCHS, batch_size=64, lr=1e-2,
                        lambda_iv=0.0, lambda_arb=1.0)
        S = float(feats["S"][0])
        K_pen = S * torch.exp(torch.linspace(-0.3, 0.3, 21, dtype=torch.float32))
        T_pen = torch.tensor(sorted({round(float(t), 6) for t in feats["T"].tolist()}),
                             dtype=torch.float32)
        train_pricer(model, feats, cfg,
                     soft_penalty_grid={"K_grid": K_pen, "T_grid": T_pen}, verbose=False)
    elif name == "bs":
        model = BlackScholesPricer()
        cfg = RunConfig(seed=0, n_epochs=min(N_EPOCHS, 80), batch_size=64, lr=2e-2,
                        lambda_iv=0.0)
        train_pricer(model, feats, cfg, verbose=False)
    else:
        raise ValueError(name)
    return model


def main():
    set_seed(0)
    out = {"probe": "full static-arbitrage compliance", "days": []}
    for di, ds in enumerate(DATES):
        date = pd.Timestamp(ds)
        snap = load_snapshot(date).call_subset()
        if not (np.isfinite(snap.risk_free_rate) and np.isfinite(snap.dividend_yield)):
            print(f"[{ds}] skip: non-finite r/q")
            continue
        filtered, _ = apply_quality_filters(snap, FilterConfig())
        feats = build_features(filtered)
        feats["iv"] = implied_vol_newton(feats["price"], feats["S"], feats["K"],
                                         feats["T"], feats["r"], feats.get("q"))
        T_grid = torch.tensor(sorted({round(float(t), 6) for t in feats["T"].tolist()}),
                              dtype=torch.float64)
        day = {"date": ds, "n_options": len(filtered), "spot": float(snap.spot),
               "r": float(snap.risk_free_rate), "q": float(snap.dividend_yield),
               "models": {}}
        print(f"[{ds}] n={len(filtered)} S={snap.spot:.1f} r={snap.risk_free_rate:.4f} "
              f"q={snap.dividend_yield:.4f} T_grid={[round(float(t),4) for t in T_grid]}")
        for name in MODELS:
            t0 = time.time()
            model = _fit(name, feats, seed=di)
            rep = full_compliance_report(
                model, S=float(snap.spot), r=float(snap.risk_free_rate),
                q=float(snap.dividend_yield), T_grid=T_grid,
            )
            dt = time.time() - t0
            print(f"  -- {name} ({dt:.0f}s)")
            print("  " + str(rep).replace("\n", "\n  "))
            d = rep.as_record()
            d["per_maturity"] = [
                {"T": m.T, "mono": m.mono_counts, "digital": m.digital_counts,
                 "upper": m.upper_counts,
                 "first_bite_K_over_S": m.mono_first_bite_K_over_S,
                 "boundary_gap_rel": m.boundary_gap_rel,
                 "left_edge_mass": m.left_edge_mass,
                 "right_edge_fwd_slope": m.right_edge_fwd_slope}
                for m in rep.per_maturity
            ]
            d["clean_in_band"] = rep.clean_in_band
            d["clean_everywhere"] = rep.clean_everywhere
            day["models"][name] = d
        out["days"].append(day)
    os.makedirs("results", exist_ok=True)
    with open("results/probe_full_compliance.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nWrote results/probe_full_compliance.json")


if __name__ == "__main__":
    main()
