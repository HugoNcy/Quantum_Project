"""Evaluation figures: MMD curve and target vs generated scatter."""

import argparse
import json
from pathlib import Path

import matplotlib

# Non-interactive backend: figures go to PNG files, never to a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import DATASETS
from model import PhotonicGenerator


def evaluate(run_dir: str, fig_dir: str = "figures", n_samples: int = 2000) -> None:
    # Load the run configuration and training history written by train.py.
    run = Path(run_dir)
    log = json.loads((run / "log.json").read_text())

    # Rebuild the generator with the same sizes as the training run.
    # Caveat (review F3): no noise argument here, so a run trained under a
    # noise profile would be silently evaluated clean; the phase 2 scripts
    # store the profile in the log and rebuild the noise model instead.
    model = PhotonicGenerator(
        latent_dim=log["latent_dim"], n_modes=log["n_modes"], n_photons=log["n_photons"]
    )
    # weights_only avoids unpickling arbitrary objects from the checkpoint.
    model.load_state_dict(torch.load(run / "model.pt", weights_only=True))
    model.eval()

    # Generate a fresh evaluation batch with a seed distinct from training.
    with torch.no_grad():
        gen = torch.Generator().manual_seed(log["seed"] + 1)
        fake = model(n_samples, generator=gen)
    # Fresh target sample with the same evaluation seed.
    real = DATASETS[log["dataset"]](n_samples, seed=log["seed"] + 1)

    figs = Path(fig_dir)
    figs.mkdir(parents=True, exist_ok=True)

    # Figure 1: training MMD curve with the real vs real floor for reference.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(log["history"], label="MMD (train)")
    ax.axhline(log["mmd_floor"], color="gray", linestyle="--", label="real vs real floor")
    ax.set_xlabel("step")
    ax.set_ylabel("MMD^2")
    # Log scale makes the approach to the floor readable.
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / f"mmd_curve_{log['dataset']}.png", dpi=150)

    # Figure 2: overlay of target and generated point clouds.
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(real[:, 0], real[:, 1], s=4, alpha=0.3, label="target")
    ax.scatter(fake[:, 0], fake[:, 1], s=4, alpha=0.3, label="generated")
    # Equal aspect so the 2D shapes are not distorted.
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
