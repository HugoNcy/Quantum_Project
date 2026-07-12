"""Phase 1bis: generator on synthetic heavy-tailed log-returns (1D target)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import log_returns
from model import PhotonicGenerator
from train_noise_grid import train_generator

# Extreme and central quantiles compared in the tails report.
TAIL_QUANTILES = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)


def run_financial(
    steps: int = 800,
    batch_size: int = 256,
    lr: float = 5e-3,
    latent_dim: int = 6,
    n_modes: int = 6,
    n_photons: int = 3,
    seed: int = 0,
    out_dir: str = "runs/financial",
    fig_dir: str = "figures",
    n_eval: int = 20000,
) -> dict:
    torch.manual_seed(seed)
    # Training pool sized like the phase 2 runs (floor pairs fit disjointly).
    pool = log_returns(20000, seed=seed)
    # Same hybrid generator as phase 1/2, clean profile, 1D output head.
    model = PhotonicGenerator(
        latent_dim=latent_dim, n_modes=n_modes, n_photons=n_photons, out_dim=1,
    )
    meta = {
        "model": "photonic",
        "dataset": "log_returns",
        "profile": "P0",
        "profile_params": {},
        "latent_dim": latent_dim, "n_modes": n_modes, "n_photons": n_photons,
        # Recorded because build_generator_from_log assumes out_dim 2; a
        # financial run is rebuilt manually with out_dim taken from the log.
        "out_dim": 1,
        "seed": seed,
    }
    log = train_generator(
        model, pool, steps=steps, batch_size=batch_size, lr=lr,
        out_dir=out_dir, meta=meta,
    )

    # Fresh evaluation samples, seed distinct from every training draw.
    model.eval()
    with torch.no_grad():
        gen = torch.Generator().manual_seed(seed + 1)
        fake = model(n_eval, generator=gen).squeeze(1)
    real = log_returns(n_eval, seed=seed + 1).squeeze(1)

    figs = Path(fig_dir)
    figs.mkdir(parents=True, exist_ok=True)

    # Histogram on a log density scale: the tails are the whole point here,
    # a linear scale would only show the (easy) central bulk.
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = torch.linspace(
        min(real.min(), fake.min()), max(real.max(), fake.max()), 120
    ).numpy()
    ax.hist(real.numpy(), bins=bins, density=True, alpha=0.5, label="target (student-t)")
    ax.hist(fake.numpy(), bins=bins, density=True, alpha=0.5, label="generated")
    ax.set_yscale("log")
    ax.set_xlabel("standardized log-return")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "hist_log_returns.png", dpi=150)

    # QQ plot: sorted generated against sorted real quantiles; identity
    # line as reference. Tail mismatch shows as departure at the ends.
    q = torch.linspace(0.001, 0.999, 399, dtype=torch.float64)
    rq = torch.quantile(real.double(), q)
    fq = torch.quantile(fake.double(), q)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rq, fq, ".", markersize=3)
    lim = float(max(rq.abs().max(), fq.abs().max()))
    ax.plot([-lim, lim], [-lim, lim], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("target quantiles")
    ax.set_ylabel("generated quantiles")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(figs / "qq_log_returns.png", dpi=150)

    # Numeric tail report used by the notebook and the write-up.
    lines = ["quantile,target,generated"]
    for p in TAIL_QUANTILES:
        rv = float(torch.quantile(real.double(), p))
        fv = float(torch.quantile(fake.double(), p))
        lines.append(f"{p},{rv:.6f},{fv:.6f}")
    (figs / "tails_log_returns.csv").write_text("\n".join(lines) + "\n")

    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="runs/financial")
    parser.add_argument("--fig-dir", default="figures")
    args = parser.parse_args()
    run_financial(steps=args.steps, seed=args.seed, out_dir=args.out_dir,
                  fig_dir=args.fig_dir)
