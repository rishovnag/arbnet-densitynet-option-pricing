#!/usr/bin/env python3
"""Mixing-dispersion sweep for the fixed-mean DensityNet (paper Section 9.7).

Traces the trade-off that Corollary (cost of a constant skew clock) predicts:
with the means fixed in maturity (h == 1), the mixing dispersion
V_m = Var_w(log m_bar) is simultaneously the model's skew budget and, through
the additive offset V_m in Var(log x_T), the source of a short-dated
at-the-money implied-volatility floor of order sqrt(V_m / T).

Sweeping a CONSTANT clock h == s would be vacuous -- it is a
reparameterisation of h == 1, since the optimiser rescales m_bar to absorb s
-- so we pin V_m itself (DensityNetConfig.fixed_V_m) and refit at each target.
The fitted skew clock is run alongside as the reference point.

Metrics are split into the short end (T <= 21 days, where the floor bites) and
the rest, on the same odd/even held-out strike split as the main study.

    python scripts/vm_sweep.py --stride 20 --out results/vm_sweep.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from typing import Dict, List
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arbnet.data import list_bhavcopy_dates, load_snapshot, apply_quality_filters, FilterConfig, build_features
from arbnet.data.iv import implied_vol_newton, bs_vega
from arbnet.models import DensityNet, DensityNetConfig
from arbnet.utils import RunConfig, set_seed
from arbnet.train import train_pricer
from arbnet.eval import rmse
from arbnet.arbitrage import adversarial_arbitrage_report

SHORT_T = 21.0 / 365.25


def _subset(feats: dict, idx) -> dict:
    n = feats["K"].size(0)
    return {k: (v[idx] if torch.is_tensor(v) and v.size(0) == n else v) for k, v in feats.items()}


def _split(feats: dict):
    T = feats["T"].numpy(); K = feats["K"].numpy()
    order = np.lexsort((K, T)); ranks = np.empty_like(order); ranks[order] = np.arange(len(order))
    return _subset(feats, np.where(ranks % 2 == 1)[0]), _subset(feats, np.where(ranks % 2 == 0)[0])


def _metrics(model, feats: dict, tag: str) -> Dict:
    with torch.no_grad():
        out = model(feats["K"], feats["T"], feats["S"], feats["r"], feats.get("q"), None)
    iv_m = implied_vol_newton(out["price"], feats["S"], feats["K"], feats["T"], feats["r"], feats.get("q"))
    q_t = feats.get("q");  q_t = torch.zeros_like(feats["r"]) if q_t is None else q_t
    vega = bs_vega(feats["S"], feats["K"], feats["T"], feats["r"], feats["iv"], q_t)
    res = {}
    for lab, mask in (("all", torch.ones_like(feats["T"], dtype=torch.bool)),
                      ("short", feats["T"] <= SHORT_T),
                      ("long", feats["T"] > SHORT_T)):
        if int(mask.sum()) == 0:
            res[f"{tag}price_{lab}"] = float("nan"); res[f"{tag}iv_{lab}"] = float("nan")
            res[f"{tag}vega_{lab}"] = float("nan"); continue
        res[f"{tag}price_{lab}"] = float(rmse(out["price"][mask], feats["price"][mask]))
        v = ~torch.isnan(iv_m) & ~torch.isnan(feats["iv"]) & mask
        res[f"{tag}iv_{lab}"] = (float(((iv_m[v] - feats["iv"][v]) ** 2).mean().sqrt()) if v.any() else float("nan"))
        ok = torch.isfinite(vega) & torch.isfinite(feats["iv"]) & (vega > 1.0) & mask
        res[f"{tag}vega_{lab}"] = (float((((out["price"][ok] - feats["price"][ok]) / vega[ok]) ** 2).mean().sqrt())
                                   if ok.any() else float("nan"))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--n_epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/vm_sweep.json")
    p.add_argument("--targets", type=float, nargs="+",
                   default=[0.0, 1e-4, 1e-3, 4e-3, 1e-2, 3e-2])
    a = p.parse_args()
    set_seed(a.seed)
    dates = list_bhavcopy_dates()[::a.stride]
    print(f"{len(dates)} days (stride {a.stride}); targets {a.targets}")
    recs: List[Dict] = []
    t0 = time.time()
    for di, date in enumerate(dates):
        try:
            snap = load_snapshot(date).call_subset()
        except Exception:
            continue
        if not (np.isfinite(snap.risk_free_rate) and np.isfinite(snap.dividend_yield)):
            continue
        filt, _ = apply_quality_filters(snap, FilterConfig())
        if len(filt) < 30:
            continue
        feats = build_features(filt)
        feats["iv"] = implied_vol_newton(feats["price"], feats["S"], feats["K"],
                                         feats["T"], feats["r"], feats.get("q"))
        tr, ev = _split(feats)
        variants = [("skew", dict(skew_clock=True))] + \
                   [(f"fixed_{t:g}", dict(fixed_V_m=t)) for t in a.targets] + \
                   [("fixed_free", dict())]
        line = [f"  [{date.date()}] n={len(filt):3d}"]
        for name, kw in variants:
            set_seed(a.seed + di)
            model = DensityNet(DensityNetConfig(context_dim=0, n_components=8, **kw))
            train_pricer(model, tr, RunConfig(seed=a.seed + di, n_epochs=a.n_epochs,
                                              batch_size=64, lr=1e-2, lambda_iv=0.0), verbose=False)
            r = {"date": str(date.date()), "variant": name}
            r.update(_metrics(model, ev, ""))
            r.update(_metrics(model, tr, "is_"))
            r.update({f"diag_{k}": v for k, v in model.mixture_diagnostics(None).items()})
            rep = adversarial_arbitrage_report(model, S=float(snap.spot), r=float(snap.risk_free_rate),
                                               q=float(snap.dividend_yield), context=None,
                                               n_base=200, n_refine=200)
            r["adv_viol"] = int(rep.butterfly_count + rep.calendar_count + rep.a4_count)
            recs.append(r)
            line.append(f" {name}:iv_s={r['iv_short']:.3f}")
        print("".join(line))
    agg: Dict[str, Dict] = {}
    for name in sorted({r["variant"] for r in recs}):
        rs = [r for r in recs if r["variant"] == name]
        agg[name] = {"n_days": len(rs), "adv_viol_days": int(sum(1 for r in rs if r["adv_viol"] > 0))}
        for k in rs[0]:
            if k in ("date", "variant", "adv_viol"):
                continue
            agg[name][k] = float(np.nanmean([r[k] for r in rs]))
    out = {"study": "DensityNet mixing-dispersion sweep (fixed means, h == 1)",
           "params": vars(a), "wall_time": time.time() - t0,
           "aggregate": agg, "records": recs}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=str)
    print(f"\nWrote {a.out} ({time.time()-t0:.0f}s)\n")
    hdr = f"{'variant':13} {'V_m':>9} {'iv_short':>9} {'iv_long':>8} {'vega_short':>11} {'price_all':>10} {'viol':>5}"
    print(hdr)
    for name, v in agg.items():
        print(f"{name:13} {v['diag_V_m']:9.2e} {v['iv_short']:9.4f} {v['iv_long']:8.4f} "
              f"{v['vega_short']:11.4f} {v['price_all']:10.2f} {v['adv_viol_days']:5d}")


if __name__ == "__main__":
    main()
