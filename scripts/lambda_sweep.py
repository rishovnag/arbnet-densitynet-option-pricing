#!/usr/bin/env python3
"""Way-forward #2: the Ackerer soft-penalty lambda-sweep (referee-critical).

Trains the soft-penalty baseline across several decades of the no-arbitrage
penalty weight lambda and traces its OWN fit-vs-violation Pareto frontier. This
answers the single most damaging omission a referee will raise: is ArbNet's
zero-violation property cheap to buy with a soft penalty, or not?

Two regimes for the baseline, selectable:
  --matched   capacity- and context-matched Ackerer (context_dim = 10 macro/aux
              features, hidden (128,128,128) ~ 35k params, ~ ArbNet's 33-43k).
              This is the fair competitor for the "gap is the constraint, not
              capacity" claim. (default)
  --plain     the original context-free (64,64,64) ~9k-param Ackerer.

For each lambda it reports mean price RMSE, mean butterfly/calendar violation
rate, and the number of days with ANY guaranteed-condition violation -- directly
comparable to ArbNet (0 / N) and to the headline Ackerer (lambda=1) result.

Cheap by design: subsample the 1359 days with --stride (default 20 -> ~68 days)
or cap with --max_days. Cost ~ (n_days * n_lambdas) single-snapshot fits.

Examples
--------
    # ~68 days, 6 lambdas, matched baseline (a couple of CPU-hours):
    python scripts/lambda_sweep.py --stride 20

    # quick smoke (10 days, 4 lambdas):
    python scripts/lambda_sweep.py --max_days 10 --lambdas 0 1 100 10000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for train_nse helpers

from arbnet.data import (
    list_bhavcopy_dates, load_snapshot, apply_quality_filters, FilterConfig,
    build_features,
)
from arbnet.data.iv import implied_vol_newton
from arbnet.models import AckererSoftPenaltyNet
from arbnet.utils import RunConfig, set_seed
from arbnet.train import train_pricer

# Reuse the exact context builder and float64 evaluator from the main study so
# numbers are directly comparable to results/nse_study_full.json.
from train_nse import build_context_frame, _evaluate


def _parse_date(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(s, "%Y-%m-%d"))


def run(dates, lambdas, n_epochs, min_options, matched, out_path, seed):
    set_seed(seed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    ctx_norm, ctx_features, _ = build_context_frame(dates) if matched else (pd.DataFrame(), [], {})
    ctx_dim = len(ctx_features)
    print(f"lambda-sweep: {len(dates)} days x {len(lambdas)} lambdas | "
          f"baseline={'matched(ctx=%d,128x3)' % ctx_dim if matched else 'plain(64x3)'}")

    # records[lam] = list of per-day dicts
    records: Dict[float, List[Dict]] = {float(l): [] for l in lambdas}
    t0 = time.time()
    for di, date in enumerate(dates):
        try:
            snap = load_snapshot(date).call_subset()
        except Exception as e:
            print(f"  [{date.date()}] SKIP load: {e}"); continue
        filtered, _ = apply_quality_filters(snap, FilterConfig())
        if len(filtered) < min_options:
            print(f"  [{date.date()}] SKIP {len(filtered)}<{min_options}"); continue
        n = len(filtered)
        feats = build_features(filtered)
        feats["iv"] = implied_vol_newton(feats["price"], feats["S"], feats["K"],
                                         feats["T"], feats.get("q"))
        ctx_vec = None
        if matched and ctx_dim > 0:
            ctx_vec = ctx_norm.loc[str(date.date()), ctx_features].to_numpy(dtype=np.float32)
            feats["context"] = torch.tensor(np.tile(ctx_vec, (n, 1)), dtype=torch.float32)

        K_pen = float(filtered.spot) * torch.exp(torch.linspace(-0.3, 0.3, 21, dtype=torch.float32))
        T_pen = torch.tensor(sorted({round(float(t), 6) for t in feats["T"].tolist()}),
                             dtype=torch.float32)

        line = [f"  [{date.date()}] n={n:4d} |"]
        for lam in lambdas:
            model = (AckererSoftPenaltyNet(context_dim=ctx_dim, hidden_dims=(128, 128, 128))
                     if matched else AckererSoftPenaltyNet())
            cfg = RunConfig(seed=seed + di, n_epochs=n_epochs, batch_size=64, lr=1e-2,
                            lambda_iv=0.0, lambda_arb=float(lam))
            # lambda=0 -> no penalty grid (pure fit); else pass the grid
            grid = {"K_grid": K_pen, "T_grid": T_pen} if lam > 0 else None
            try:
                train_pricer(model, feats, cfg, soft_penalty_grid=grid, verbose=False)
                rec = _evaluate(model, feats, filtered,
                                f"ackerer_lam{lam:g}", ctx_vec if matched else None)
            except Exception as e:
                print(f"  [{date.date()}] lam={lam} FAILED: {e}"); continue
            rec["date"] = str(date.date()); rec["lambda"] = float(lam)
            records[float(lam)].append(rec)
            line.append(f" l{lam:g}:RMSE={rec['price_rmse']:6.1f} "
                        f"bf={rec['butterfly_rate']:.1%}")
        print("".join(line))

    wall = time.time() - t0

    # --- Pareto aggregate per lambda ------------------------------------------
    pareto = []
    for lam in lambdas:
        recs = records[float(lam)]
        if not recs:
            continue
        bf = np.array([r["butterfly_rate"] for r in recs])
        cal = np.array([r["calendar_rate"] for r in recs])
        pareto.append({
            "lambda": float(lam),
            "n_days": len(recs),
            "price_rmse_mean": float(np.mean([r["price_rmse"] for r in recs])),
            "price_rmse_std": float(np.std([r["price_rmse"] for r in recs])),
            "butterfly_rate_mean": float(bf.mean()),
            "butterfly_rate_max": float(bf.max()),
            "calendar_rate_mean": float(cal.mean()),
            "calendar_rate_max": float(cal.max()),
            "violation_days": int(np.sum((bf > 0) | (cal > 0))),
        })

    summary = {
        "study": "Ackerer soft-penalty lambda sweep (fit-vs-violation Pareto)",
        "baseline": "matched" if matched else "plain",
        "context_dim": ctx_dim,
        "params": {"n_days": len(dates), "lambdas": [float(l) for l in lambdas],
                   "n_epochs": n_epochs, "seed": seed},
        "wall_time": wall,
        "pareto": pareto,
        "records": {f"{l:g}": records[float(l)] for l in lambdas},
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote {out_path}  ({wall:.1f}s)\n")
    print("=== Ackerer fit-vs-violation Pareto ===")
    print(f"{'lambda':>10}{'RMSE':>10}{'butterfly':>12}{'calendar':>12}{'viol_days':>11}")
    for p in pareto:
        print(f"{p['lambda']:>10g}{p['price_rmse_mean']:>10.1f}"
              f"{p['butterfly_rate_mean']:>11.3%}{p['calendar_rate_mean']:>11.3%}"
              f"{p['violation_days']:>8d}/{p['n_days']:<3d}")
    print("\nCompare to ArbNet: RMSE ~95 (real), 0 butterfly / 0 calendar / 0 viol_days.")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", type=_parse_date, default=None)
    p.add_argument("--end", type=_parse_date, default=None)
    p.add_argument("--stride", type=int, default=20, help="use every Nth trading day")
    p.add_argument("--max_days", type=int, default=0)
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[0.0, 0.1, 1.0, 10.0, 100.0, 1000.0])
    p.add_argument("--n_epochs", type=int, default=150)
    p.add_argument("--min_options", type=int, default=30)
    p.add_argument("--plain", action="store_true",
                   help="use the original context-free (64,64,64) Ackerer instead of matched")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/lambda_sweep.json")
    args = p.parse_args()

    all_dates = list_bhavcopy_dates()
    if not all_dates:
        print("No bhavcopy files found; run scripts/check_data.py."); sys.exit(1)
    dates = all_dates
    if args.start is not None:
        dates = [d for d in dates if d >= args.start]
    if args.end is not None:
        dates = [d for d in dates if d <= args.end]
    dates = dates[:: max(1, args.stride)]
    if args.max_days and args.max_days > 0:
        dates = dates[: args.max_days]
    if not dates:
        print("No trading days selected."); sys.exit(1)

    run(dates, args.lambdas, args.n_epochs, args.min_options,
        matched=not args.plain, out_path=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
