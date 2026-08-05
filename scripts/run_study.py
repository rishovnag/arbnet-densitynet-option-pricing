"""Synthetic experimental study for the ArbNet paper.

Runs ArbNet vs AckererSoftPenaltyNet (and BS baseline) on a population of
random rough-Bergomi surfaces. Reports:
- price RMSE, IV RMSE
- arbitrage violation rates (butterfly, calendar, tail)
- hedging PnL std and CVaR95
- statistical significance (plain paired t-test on per-(surface, seed) RMSE;
  uncorrected -- the real study's HAC Diebold-Mariano is the rigorous one)

Results saved to results/study.json for the paper tables.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbnet.data import (
    RoughBergomiGenerator,
    apply_quality_filters,
    FilterConfig,
    build_features,
)
from arbnet.data.iv import implied_vol_newton
from arbnet.models import (ArbNet, AckererSoftPenaltyNet, BlackScholesPricer,
                           RoughBergomiSimulator, DensityNet, DensityNetConfig)
from arbnet.utils import default_arbnet_config, RunConfig, set_seed
from arbnet.train import train_pricer
from arbnet.arbitrage import static_arbitrage_report, adversarial_arbitrage_report
from arbnet.eval import rmse, hedge_pnl_stats
from arbnet.hedging import hedge_pnl_delta


# -----------------------------------------------------------------------------
def _grid_eval(model, K_grid, T_grid, S0, r, q):
    """Evaluate a model on a (T, K) grid in float64.

    The arbitrage report differences grid prices against a fixed 1e-7
    tolerance, which is only "one decade above noise" in double precision.
    In float32, Nifty-scale prices (S = 20000 here) carry ~1e-4 differencing
    noise -- three decades above tolerance -- which manufactures spurious
    butterfly/calendar violations. So the grid is evaluated in float64.
    """
    n_T, n_K = T_grid.shape[0], K_grid.shape[0]
    Tm, Km = torch.meshgrid(T_grid.double(), K_grid.double(), indexing="ij")
    Tf, Kf = Tm.flatten(), Km.flatten()
    S_b = torch.full_like(Tf, float(S0))
    r_b = torch.full_like(Tf, float(r))
    q_b = torch.full_like(Tf, float(q))
    model.double()
    try:
        with torch.no_grad():
            out = model(Kf, Tf, S_b, r_b, q_b)
            if "w" in out and out["w"].abs().max() > 0:
                w = out["w"]
            else:
                iv = implied_vol_newton(out["price"], S_b, Kf, Tf, r_b, q_b)
                w = (iv * iv * Tf).nan_to_num(nan=0.0)
            price = out["price"]
    finally:
        model.float()   # restore float32 for training / price-RMSE / hedging
    return price.view(n_T, n_K), w.view(n_T, n_K)


def _evaluate_one(model, feats, K_grid, T_grid, snap, name: str) -> Dict:
    """Evaluate a trained model on in-sample fit and arbitrage compliance."""
    with torch.no_grad():
        out = model(feats["K"], feats["T"], feats["S"], feats["r"], feats.get("q"))
    price_rmse = float(rmse(out["price"], feats["price"]))
    # IV RMSE: compare model IV (inverted from price) to data IV (already inverted)
    iv_model = implied_vol_newton(
        out["price"], feats["S"], feats["K"], feats["T"], feats["r"], feats.get("q"),
    )
    if "iv" in feats:
        valid = ~torch.isnan(iv_model) & ~torch.isnan(feats["iv"])
        iv_rmse = float(((iv_model[valid] - feats["iv"][valid]) ** 2).mean().sqrt().item()) if valid.any() else float("nan")
    else:
        iv_rmse = float("nan")
    # Arbitrage on a (T, K) grid
    k_grid = torch.log(K_grid / float(snap.spot))
    C_grid, w_grid = _grid_eval(model, K_grid, T_grid, snap.spot, snap.risk_free_rate, snap.dividend_yield)
    rep = static_arbitrage_report(
        C_grid=C_grid, w_grid=w_grid, K_grid=K_grid, T_grid=T_grid, k_grid=k_grid,
        r=float(snap.risk_free_rate), q=float(snap.dividend_yield),
    )
    return {
        "model": name,
        "price_rmse": price_rmse,
        "iv_rmse": iv_rmse,
        "butterfly_count": rep.butterfly_count,
        "butterfly_rate": rep.butterfly_rate,
        "calendar_count": rep.calendar_count,
        "calendar_rate": rep.calendar_rate,
        "tail_count": rep.tail_count,
        "tail_rate": rep.tail_rate,
        "total_rate": rep.total_rate,
    }


def _hedge_eval(model, snap, T_h: float = 30 / 365.0, n_paths: int = 500, seed: int = 0) -> Dict:
    """Simulate hedging PnL using the model's delta. Returns std + CVaR95."""
    sim = RoughBergomiSimulator(H=0.10, eta=1.5, rho=-0.7, xi0=0.04, dtype=torch.float64)
    t_grid = torch.linspace(0.0, T_h, 16, dtype=torch.float64)
    S_paths, _ = sim.simulate(n_paths, t_grid, S0=snap.spot, r=snap.risk_free_rate, q=snap.dividend_yield, seed=seed)
    K_h = torch.tensor(float(snap.spot))
    with torch.no_grad():
        if hasattr(model, "implied_vol"):
            iv = model.implied_vol(
                torch.tensor([float(snap.spot)]),
                torch.tensor([T_h]),
                torch.tensor([float(snap.spot)]),
                torch.tensor([float(snap.risk_free_rate)]),
                torch.tensor([float(snap.dividend_yield)]),
            )
            sig = float(iv.item()) if not torch.isnan(iv).any() else 0.20
        else:
            out_atm = model(
                torch.tensor([float(snap.spot)]),
                torch.tensor([T_h]),
                torch.tensor([float(snap.spot)]),
                torch.tensor([float(snap.risk_free_rate)]),
                torch.tensor([float(snap.dividend_yield)]),
            )
            if "iv" in out_atm and not torch.isnan(out_atm["iv"]).any():
                sig = float(out_atm["iv"].item())
            else:
                sig = 0.20
    pnl = hedge_pnl_delta(
        S_paths.float(), K_h, torch.tensor(float(T_h)),
        torch.tensor(float(snap.risk_free_rate)),
        torch.tensor(sig), dt=float(T_h / 15), cost_bps=2.0,
    )
    stats = hedge_pnl_stats(pnl)
    return {"hedge_std": float(stats.std), "hedge_cvar95": float(stats.cvar_95)}


# -----------------------------------------------------------------------------
def run_study(
    n_surfaces: int = 20,
    seeds_per_surface: int = 3,
    n_epochs: int = 150,
    out_path: str = "results/study.json",
    models: List[str] = ("ArbNet", "Ackerer", "BS"),
    resume: bool = False,
) -> Dict:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    models = list(models)

    # Resume/merge: never retrain a (surface, seed, model) already in the file.
    existing_records: List[Dict] = []
    if resume and os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            existing_records = prev.get("records", [])
            print(f"Resume: loaded {len(existing_records)} existing records from "
                  f"{out_path}; those (surface, seed, model) will NOT be retrained.")
        except Exception as e:
            print(f"Resume: could not read {out_path} ({e}); starting fresh.")
    have = {(r.get("surface"), r.get("seed"), r.get("model")) for r in existing_records}

    all_records: List[Dict] = list(existing_records)
    summary: Dict = {"surfaces": [], "params": {
        "n_surfaces": n_surfaces,
        "seeds_per_surface": seeds_per_surface,
        "n_epochs": n_epochs,
        "models": models,
        "resume": resume,
    }}
    t0 = time.time()
    # Sample rough-Bergomi parameter regimes (Hurst H, vol-of-vol eta, leverage rho)
    rng = np.random.default_rng(0)
    surfaces_params = []
    for s_idx in range(n_surfaces):
        H = float(rng.uniform(0.07, 0.20))
        eta = float(rng.uniform(1.0, 3.0))
        rho = float(rng.uniform(-0.9, -0.3))
        xi0 = float(rng.uniform(0.02, 0.10))
        # NB: key is "surface_seed" (the surface-generation seed), kept distinct
        # from the per-trial model-init "seed" so the two never collide when
        # this dict is splatted into a record below.
        surfaces_params.append({"H": H, "eta": eta, "rho": rho, "xi0": xi0,
                                "surface_seed": s_idx})

    for s_idx, params in enumerate(surfaces_params):
        print(f"\n=== Surface {s_idx + 1}/{n_surfaces}  H={params['H']:.3f}  eta={params['eta']:.2f}  rho={params['rho']:.2f}  xi0={params['xi0']:.3f} ===")
        # Generate rough-Bergomi surface
        gen = RoughBergomiGenerator(
            H=params["H"], eta=params["eta"], rho=params["rho"], xi0=params["xi0"],
            n_paths=4000, seed=params["surface_seed"],
        )
        snap = gen.generate(n_strikes=20, maturities_days=[7, 14, 30, 60, 90])
        filtered, summ = apply_quality_filters(snap, FilterConfig())
        if len(filtered) < 30:
            print(f"  too few options after filter ({len(filtered)}); skipping")
            continue
        feats = build_features(filtered)

        # Grids for the arbitrage / soft-penalty evaluation
        K_grid_pen = filtered.spot * torch.exp(torch.linspace(-0.3, 0.3, 21, dtype=torch.float32))
        K_grid_eval = filtered.spot * torch.exp(torch.linspace(-0.35, 0.35, 41, dtype=torch.float32))
        T_grid = torch.tensor([7 / 365, 14 / 365, 30 / 365, 60 / 365, 90 / 365], dtype=torch.float32)

        for seed in range(seeds_per_surface):
            msg = {}   # model -> short status string for the per-seed print

            def _todo(model_name: str) -> bool:
                return model_name in models and (s_idx, seed, model_name) not in have

            def _store(rec: Dict) -> None:
                # "seed" (per-trial model init) last so it cannot be clobbered by **params.
                rec.update({"surface": s_idx, **params, "seed": seed})
                all_records.append(rec)

            # ArbNet -- lr=1e-2 (we need to fit absolute Nifty-scale prices)
            if _todo("ArbNet"):
                arbnet = ArbNet(default_arbnet_config())
                cfg_a = RunConfig(seed=seed, n_epochs=n_epochs, batch_size=64, lr=1e-2, lambda_iv=0.0)
                train_pricer(arbnet, feats, cfg_a, verbose=False)
                rec_a = _evaluate_one(arbnet, feats, K_grid_eval, T_grid, filtered, name="ArbNet")
                rec_a.update(_hedge_eval(arbnet, filtered, seed=seed))
                _store(rec_a)
                msg["ArbNet"] = f"ArbNet RMSE={rec_a['price_rmse']:.2f} viol={rec_a['butterfly_rate']:.3%}/{rec_a['calendar_rate']:.3%}"

            # ArbNet + concave bump (A3/A4/mass by construction; butterfly/calendar monitored)
            if _todo("ArbNet_bump"):
                cfg_net = default_arbnet_config()
                cfg_net.use_concave_bump = True
                arbnet_b = ArbNet(cfg_net)
                cfg_ab = RunConfig(seed=seed, n_epochs=n_epochs, batch_size=64, lr=1e-2, lambda_iv=0.0)
                train_pricer(arbnet_b, feats, cfg_ab, verbose=False)
                rec_ab = _evaluate_one(arbnet_b, feats, K_grid_eval, T_grid, filtered, name="ArbNet_bump")
                adv = adversarial_arbitrage_report(
                    arbnet_b, S=float(filtered.spot), r=float(filtered.risk_free_rate),
                    q=float(filtered.dividend_yield), T_grid=T_grid.double(),
                    n_base=400, n_refine=400,
                )
                rec_ab.update({
                    "adv_butterfly_count": adv.butterfly_count,
                    "adv_calendar_count": adv.calendar_count,
                    "adv_a4_count": adv.a4_count,
                    "adv_butterfly_worst": adv.butterfly_worst,
                })
                rec_ab.update(_hedge_eval(arbnet_b, filtered, seed=seed))
                _store(rec_ab)
                msg["ArbNet_bump"] = (f"ArbNet_bump RMSE={rec_ab['price_rmse']:.2f} "
                                      f"adv_bfly={adv.butterfly_count}")

            # DensityNet: martingale lognormal-mixture, A1-A4 arbitrage-free by construction
            if _todo("ArbNet_density"):
                dnet = DensityNet(DensityNetConfig(context_dim=0, n_components=8))
                cfg_d = RunConfig(seed=seed, n_epochs=n_epochs, batch_size=64, lr=1e-2, lambda_iv=0.0)
                train_pricer(dnet, feats, cfg_d, verbose=False)
                rec_d = _evaluate_one(dnet, feats, K_grid_eval, T_grid, filtered, name="ArbNet_density")
                adv = adversarial_arbitrage_report(
                    dnet, S=float(filtered.spot), r=float(filtered.risk_free_rate),
                    q=float(filtered.dividend_yield), T_grid=T_grid.double(),
                    n_base=400, n_refine=400)
                rec_d.update({"adv_butterfly_count": adv.butterfly_count,
                              "adv_calendar_count": adv.calendar_count,
                              "adv_calendar_strike_count": adv.calendar_strike_count,
                              "adv_a4_count": adv.a4_count,
                              "adv_butterfly_worst": adv.butterfly_worst})
                rec_d.update(_hedge_eval(dnet, filtered, seed=seed))
                _store(rec_d)
                msg["ArbNet_density"] = (f"ArbNet_density RMSE={rec_d['price_rmse']:.2f} "
                                         f"adv_bfly={adv.butterfly_count}")

            # Ackerer capacity-matched (~35k params; no context in the synthetic study)
            if _todo("Ackerer_matched"):
                ackm = AckererSoftPenaltyNet(hidden_dims=(128, 128, 128))
                cfg_m = RunConfig(seed=seed, n_epochs=n_epochs, batch_size=64, lr=1e-2,
                                  lambda_iv=0.0, lambda_arb=1.0)
                train_pricer(ackm, feats, cfg_m,
                             soft_penalty_grid={"K_grid": K_grid_pen, "T_grid": T_grid},
                             verbose=False)
                rec_m = _evaluate_one(ackm, feats, K_grid_eval, T_grid, filtered, name="Ackerer_matched")
                rec_m.update(_hedge_eval(ackm, filtered, seed=seed))
                _store(rec_m)
                msg["Ackerer_matched"] = f"Ackerer_matched RMSE={rec_m['price_rmse']:.2f}"

            # Ackerer (soft penalty)
            if _todo("Ackerer"):
                ackerer = AckererSoftPenaltyNet()
                cfg_b = RunConfig(seed=seed, n_epochs=n_epochs, batch_size=64, lr=1e-2, lambda_iv=0.0, lambda_arb=1.0)
                grid_pen = {"K_grid": K_grid_pen, "T_grid": T_grid}
                train_pricer(ackerer, feats, cfg_b, soft_penalty_grid=grid_pen, verbose=False)
                rec_b = _evaluate_one(ackerer, feats, K_grid_eval, T_grid, filtered, name="Ackerer")
                rec_b.update(_hedge_eval(ackerer, filtered, seed=seed))
                _store(rec_b)
                msg["Ackerer"] = f"Ackerer RMSE={rec_b['price_rmse']:.2f} viol={rec_b['butterfly_rate']:.3%}/{rec_b['calendar_rate']:.3%}"

            # Black-Scholes (ATM-fitted) -- only one seed needed (no NN init)
            if _todo("BS"):
                bs = BlackScholesPricer()
                cfg_c = RunConfig(seed=0, n_epochs=80, batch_size=64, lr=2e-2, lambda_iv=0.0)
                train_pricer(bs, feats, cfg_c, verbose=False)
                rec_c = _evaluate_one(bs, feats, K_grid_eval, T_grid, filtered, name="BS")
                rec_c.update(_hedge_eval(bs, filtered, seed=seed))
                _store(rec_c)
                msg["BS"] = f"BS RMSE={rec_c['price_rmse']:.2f}"

            if msg:
                print(f"  seed {seed}: " + "  ".join(msg[m] for m in models if m in msg))

        summary["surfaces"].append({"surface": s_idx, **params, "n_options": len(filtered)})

    wall_time = time.time() - t0
    summary["wall_time"] = wall_time
    summary["n_records"] = len(all_records)
    summary["records"] = all_records

    # Aggregate: per-model means and stds (over every model present in records).
    by_model: Dict[str, List[Dict]] = {}
    for r in all_records:
        by_model.setdefault(r["model"], []).append(r)
    agg = {}
    for name, recs in by_model.items():
        if len(recs) == 0:
            continue
        agg[name] = {
            "n": len(recs),
            "price_rmse_mean": float(np.mean([r["price_rmse"] for r in recs])),
            "price_rmse_std":  float(np.std([r["price_rmse"] for r in recs])),
            "iv_rmse_mean":    float(np.nanmean([r["iv_rmse"] for r in recs])),
            "iv_rmse_std":     float(np.nanstd([r["iv_rmse"] for r in recs])),
            "butterfly_rate_mean": float(np.mean([r["butterfly_rate"] for r in recs])),
            "butterfly_rate_max":  float(np.max([r["butterfly_rate"] for r in recs])),
            "calendar_rate_mean":  float(np.mean([r["calendar_rate"] for r in recs])),
            "calendar_rate_max":   float(np.max([r["calendar_rate"] for r in recs])),
            "tail_rate_mean":      float(np.mean([r["tail_rate"] for r in recs])),
            "total_rate_mean":     float(np.mean([r["total_rate"] for r in recs])),
            "hedge_std_mean":   float(np.nanmean([r["hedge_std"] for r in recs])),
            "hedge_cvar95_mean": float(np.nanmean([r["hedge_cvar95"] for r in recs])),
        }
    summary["aggregate"] = agg

    # Paired comparisons: surfaces where ArbNet vs Ackerer.
    # A plain paired t-test on per-(surface, seed) RMSE differences (M3: this is
    # NOT a Diebold-Mariano test -- no HAC correction, and it ignores the
    # correlation between seeds of the same surface; label it accordingly).
    arbnet_by_key = {(r["surface"], r["seed"]): r for r in by_model.get("ArbNet", [])}
    ackerer_by_key = {(r["surface"], r["seed"]): r for r in by_model.get("Ackerer", [])}
    diffs = []
    for key, ra in arbnet_by_key.items():
        if key in ackerer_by_key:
            rb = ackerer_by_key[key]
            diffs.append(ra["price_rmse"] - rb["price_rmse"])
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) >= 5:
        mean_diff = float(diffs.mean())
        sd = float(diffs.std(ddof=1))
        n = len(diffs)
        t_stat = mean_diff / (sd / math.sqrt(n)) if sd > 0 else float("nan")
        summary["paired_test"] = {
            "test": "paired t-test (uncorrected; not a DM test)",
            "n_pairs": n,
            "mean_diff_arbnet_minus_ackerer": mean_diff,
            "std_diff": sd,
            "t_stat": t_stat,
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_path}  ({wall_time:.1f}s, {len(all_records)} records)")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_surfaces", type=int, default=20)
    p.add_argument("--seeds_per_surface", type=int, default=3)
    p.add_argument("--n_epochs", type=int, default=150)
    p.add_argument("--out", type=str, default="results/study.json")
    p.add_argument("--models", nargs="+", default=["ArbNet", "Ackerer", "BS"],
                   choices=["ArbNet", "Ackerer", "BS", "ArbNet_bump", "Ackerer_matched", "ArbNet_density"])
    p.add_argument("--resume", action="store_true",
                   help="merge into an existing --out file: reuse already-computed "
                        "(surface, seed, model) records and only train missing models")
    args = p.parse_args()
    set_seed(0)
    summary = run_study(
        n_surfaces=args.n_surfaces,
        seeds_per_surface=args.seeds_per_surface,
        n_epochs=args.n_epochs,
        out_path=args.out,
        models=args.models,
        resume=args.resume,
    )
    print("\n=== Aggregate ===")
    for name, a in summary["aggregate"].items():
        print(f"  {name}: n={a['n']}  RMSE={a['price_rmse_mean']:.2f} ± {a['price_rmse_std']:.2f}  "
              f"viol_butt={a['butterfly_rate_mean']:.3%}  viol_cal={a['calendar_rate_mean']:.3%}  "
              f"hedge_std={a['hedge_std_mean']:.2f}")


if __name__ == "__main__":
    main()
