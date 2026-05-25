"""End-to-end smoke test.

Runs the entire pipeline (data generation -> training -> evaluation -> arbitrage
check -> hedging) on a small synthetic surface in a few seconds. Exits with
non-zero status if anything fails. Use this as a CI sanity check before larger
runs.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arbnet.data import (
    SyntheticSurfaceGenerator,
    apply_quality_filters,
    FilterConfig,
    build_features,
)
from arbnet.models import ArbNet, AckererSoftPenaltyNet, RoughBergomiSimulator
from arbnet.utils import default_arbnet_config, RunConfig, set_seed
from arbnet.train import train_pricer
from arbnet.arbitrage import static_arbitrage_report
from arbnet.eval import rmse, hedge_pnl_stats
from arbnet.hedging import hedge_pnl_delta
from arbnet.data.iv import implied_vol_newton


def step(name, fn):
    print(f"\n--- {name} ---")
    try:
        result = fn()
        print(f"OK: {name}")
        return result
    except Exception as e:
        print(f"FAIL: {name}: {e}")
        traceback.print_exc()
        sys.exit(1)


def _grid_eval(model, K_grid, T_grid, S0, r, q):
    """Evaluate model on a (T, K) grid; return (C_grid, w_grid)."""
    n_T, n_K = T_grid.shape[0], K_grid.shape[0]
    Tm, Km = torch.meshgrid(T_grid, K_grid, indexing="ij")
    Tf, Kf = Tm.flatten(), Km.flatten()
    S_b = torch.full_like(Tf, float(S0))
    r_b = torch.full_like(Tf, float(r))
    q_b = torch.full_like(Tf, float(q))
    with torch.no_grad():
        out = model(Kf, Tf, S_b, r_b, q_b)
        if "w" in out and out["w"].abs().max() > 0:
            w = out["w"]
        else:
            iv = implied_vol_newton(out["price"], S_b, Kf, Tf, r_b, q_b)
            w = (iv * iv * Tf).nan_to_num(nan=0.0)
    return out["price"].view(n_T, n_K), w.view(n_T, n_K)


def main():
    set_seed(0)

    snap = step(
        "Generate synthetic surface",
        lambda: SyntheticSurfaceGenerator(seed=0).generate(
            n_strikes=15, maturities_days=[7, 14, 30, 60]
        ),
    )
    print(f"  n_options = {len(snap)}")

    filtered, summary = step(
        "Apply quality filters",
        lambda: apply_quality_filters(snap, FilterConfig()),
    )
    print(f"  kept {summary['n_kept']}/{summary['n_initial']} options")

    feats = step("Build features", lambda: build_features(filtered))
    print(f"  feature keys = {list(feats.keys())}")

    # Train ArbNet (direct call-price parametrization)
    model = ArbNet(default_arbnet_config())
    cfg = RunConfig(seed=0, n_epochs=40, batch_size=64, lr=5e-3, lambda_iv=0.0)
    step(
        "Train ArbNet (40 epochs)",
        lambda: train_pricer(model, feats, cfg, verbose=False),
    )

    def eval_arbnet():
        with torch.no_grad():
            out = model(feats["K"], feats["T"], feats["S"], feats["r"], feats.get("q"))
        return rmse(out["price"], feats["price"])

    rmse_val = step("Evaluate ArbNet price RMSE", eval_arbnet)
    print(f"  ArbNet price RMSE = {rmse_val:.4f}")

    # Train Ackerer baseline
    model_a = AckererSoftPenaltyNet()
    cfg_a = RunConfig(
        seed=0, n_epochs=40, batch_size=64, lr=5e-3, lambda_iv=0.0, lambda_arb=1.0
    )
    K_grid_pen = filtered.spot * torch.exp(torch.linspace(-0.3, 0.3, 21, dtype=torch.float32))
    T_grid_pen = torch.tensor([7 / 365, 14 / 365, 30 / 365, 60 / 365], dtype=torch.float32)
    grid_pen = {"K_grid": K_grid_pen, "T_grid": T_grid_pen}
    step(
        "Train Ackerer baseline (40 epochs)",
        lambda: train_pricer(model_a, feats, cfg_a, soft_penalty_grid=grid_pen, verbose=False),
    )

    # Arbitrage check on a grid for both models
    K_grid = filtered.spot * torch.exp(torch.linspace(-0.35, 0.35, 41, dtype=torch.float32))
    T_grid = torch.tensor([7 / 365, 14 / 365, 30 / 365, 60 / 365], dtype=torch.float32)
    k_grid = torch.log(K_grid / (filtered.spot * float(torch.exp(torch.tensor((filtered.risk_free_rate - filtered.dividend_yield) * 30 / 365.0)))))

    def arb_check(m, name):
        C_grid, w_grid = _grid_eval(
            m, K_grid, T_grid, filtered.spot, filtered.risk_free_rate, filtered.dividend_yield
        )
        rep = static_arbitrage_report(
            C_grid=C_grid, w_grid=w_grid, K_grid=K_grid, T_grid=T_grid, k_grid=k_grid,
            r=float(filtered.risk_free_rate), q=float(filtered.dividend_yield),
        )
        print(f"  [{name}] {rep}")
        return rep

    step("Arbitrage check (ArbNet)", lambda: arb_check(model, "ArbNet"))
    step("Arbitrage check (Ackerer)", lambda: arb_check(model_a, "Ackerer"))

    # Hedging simulation: invert IV from ArbNet's ATM price for delta hedging
    def hedge_check():
        sim = RoughBergomiSimulator(H=0.1, eta=1.5, rho=-0.7, xi0=0.04, dtype=torch.float64)
        T_h = 30 / 365.0
        t_grid = torch.linspace(0.0, T_h, 16, dtype=torch.float64)
        S_paths, _ = sim.simulate(
            500, t_grid, S0=filtered.spot,
            r=filtered.risk_free_rate, q=filtered.dividend_yield, seed=0,
        )
        K_h = torch.tensor(float(filtered.spot))
        with torch.no_grad():
            out_atm = model(
                torch.tensor([float(filtered.spot)]),
                torch.tensor([T_h]),
                torch.tensor([float(filtered.spot)]),
                torch.tensor([float(filtered.risk_free_rate)]),
                torch.tensor([float(filtered.dividend_yield)]),
            )
            iv_atm = model.implied_vol(
                torch.tensor([float(filtered.spot)]),
                torch.tensor([T_h]),
                torch.tensor([float(filtered.spot)]),
                torch.tensor([float(filtered.risk_free_rate)]),
                torch.tensor([float(filtered.dividend_yield)]),
            )
        sig = float(iv_atm.item()) if not torch.isnan(iv_atm).any() else 0.2
        pnl = hedge_pnl_delta(
            S_paths.float(), K_h, torch.tensor(float(T_h)),
            torch.tensor(float(filtered.risk_free_rate)),
            torch.tensor(sig), dt=float(T_h / 15), cost_bps=1.0,
        )
        return hedge_pnl_stats(pnl)

    stats = step("Hedging PnL simulation", hedge_check)
    print(f"  PnL std={stats.std:.4f}, CVaR95={stats.cvar_95:.4f}")

    print("\n=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()
