"""Evaluation figures: MMD curve and target vs generated scatter."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import DATASETS
from model import PhotonicGenerator


def evaluate(run_dir: str, fig_dir: str = "figures", n_samples: int = 2000) -> None:
    run = Path(run_dir)
    log = json.loads((run / "log.json").read_text())

    model = PhotonicGenerator(
        latent_dim=log["latent_dim"], n_modes=log["n_modes"], n_photons=log["n_photons"]
    )
    model.load_state_dict(torch.load(run / "model.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        gen = torch.Generator().manual_seed(log["seed"] + 1)
        fake = model(n_samples, generator=gen)
    real = DATASETS[log["dataset"]](n_samples, seed=log["seed"] + 1)

    figs = Path(fig_dir)
    figs.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(log["history"], label="MMD (train)")
    ax.axhline(log["mmd_floor"], color="gray", linestyle="--", label="real vs real floor")
    ax.set_xlabel("step")
    ax.set_ylabel("MMD^2")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / f"mmd_curve_{log['dataset']}.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(real[:, 0], real[:, 1], s=4, alpha=0.3, label="target")
    ax.scatter(fake[:, 0], fake[:, 1], s=4, alpha=0.3, label="generated")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / f"scatter_{log['dataset']}.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/baseline")
    parser.add_argument("--fig-dir", default="figures")
    args = parser.parse_args()
    evaluate(args.run_dir, args.fig_dir)
