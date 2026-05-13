"""End-to-end training entry point.

Example:
    python scripts/train.py --model arbnet --data synthetic --epochs 200
"""
import argparse
import os
import sys
import json

import torch

# Add parent directory to path for local execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbnet.data import (
    SyntheticSurfaceGenerator,
    RoughBergomiGenerator,
    apply_quality_filters,
    FilterConfig,
    build_features,
)
from arbnet.models import ArbNet, ArbNetConfig, AckererSoftPenaltyNet, BlackScholesPricer
from arbnet.train import train_pricer
from arbnet.utils import RunConfig, default_arbnet_config


def make_model(name: str, context_dim: int = 0):
    if name == "arbnet":
        return ArbNet(default_arbnet_config(context_dim=context_dim))
    if name == "ackerer":
        return AckererSoftPenaltyNet(context_dim=context_dim)
    if name == "bs":
        return BlackScholesPricer()
    raise ValueError(f"Unknown model {name}")


def make_data(name: str, seed: int):
    if name == "synthetic":
        gen = SyntheticSurfaceGenerator(seed=seed)
        return gen.generate()
    if name == "rough_bergomi":
        gen = RoughBergomiGenerator(seed=seed)
        return gen.generate()
    raise ValueError(f"Unknown dataset {name}. For NSE data use scripts/train_nse.py.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["arbnet", "ackerer", "bs"], default="arbnet")
    p.add_argument("--data", choices=["synthetic", "rough_bergomi"], default="synthetic")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda_arb", type=float, default=None,
                   help="Soft-arbitrage penalty (Ackerer baseline only). Defaults: 0 for arbnet/bs, 1.0 for ackerer.")
    p.add_argument("--output_dir", default="runs/")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"== Generating {args.data} surface ==")
    snap = make_data(args.data, seed=args.seed)
    filtered, _ = apply_quality_filters(snap, FilterConfig())
    features = build_features(filtered)
    print(f"  n_options after filtering: {len(filtered)}")

    print(f"== Building {args.model} ==")
    model = make_model(args.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params}")

    lambda_arb = args.lambda_arb if args.lambda_arb is not None else (1.0 if args.model == "ackerer" else 0.0)
    cfg = RunConfig(seed=args.seed, n_epochs=args.epochs, lr=args.lr, lambda_arb=lambda_arb)

    # Build a soft-penalty grid for the Ackerer baseline
    grid = None
    if args.model == "ackerer":
        grid = {
            "k_grid": torch.linspace(-0.4, 0.4, 41),
            "T_grid": torch.tensor([7/365, 14/365, 30/365, 60/365, 90/365], dtype=torch.float32),
        }

    print("== Training ==")
    diag = train_pricer(model, features, cfg, soft_penalty_grid=grid)
    print(f"== Done in {diag['wall_time']:.1f}s, final loss={diag['loss_history'][-1]:.6f} ==")

    # Save
    ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.data}_seed{args.seed}.pt")
    torch.save({"state_dict": model.state_dict(), "config": cfg.as_dict()}, ckpt_path)
    diag_path = os.path.join(args.output_dir, f"{args.model}_{args.data}_seed{args.seed}.json")
    with open(diag_path, "w") as f:
        json.dump({"loss_history": diag["loss_history"], "wall_time": diag["wall_time"]}, f)
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
