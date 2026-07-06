"""Training loop for the hybrid photonic generator with MMD loss."""

import argparse
import json
from pathlib import Path

import torch

from data import DATASETS
from losses import mmd_loss
from model import PhotonicGenerator


def train(
    dataset: str = "two_gaussians",
    steps: int = 800,
    batch_size: int = 256,
    lr: float = 5e-3,
    latent_dim: int = 6,
    n_modes: int = 6,
    n_photons: int = 3,
    seed: int = 0,
    out_dir: str = "runs/baseline",
) -> dict:
    torch.manual_seed(seed)
    target_fn = DATASETS[dataset]
    target_pool = target_fn(20000, seed=seed)

    model = PhotonicGenerator(latent_dim=latent_dim, n_modes=n_modes, n_photons=n_photons)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # MMD between two real batches gives the floor we can compare against
    floor = mmd_loss(target_pool[:batch_size], target_pool[batch_size:2 * batch_size]).item()

    history = []
    for step in range(steps):
        idx = torch.randint(0, len(target_pool), (batch_size,))
        real = target_pool[idx]
        fake = model(batch_size)
        loss = mmd_loss(fake, real)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    log = {
        "dataset": dataset, "steps": steps, "batch_size": batch_size, "lr": lr,
        "latent_dim": latent_dim, "n_modes": n_modes, "n_photons": n_photons,
        "seed": seed, "mmd_floor": floor, "history": history,
    }
    (out / "log.json").write_text(json.dumps(log))
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="two_gaussians", choices=list(DATASETS))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="runs/baseline")
    args = parser.parse_args()
    train(dataset=args.dataset, steps=args.steps, seed=args.seed, out_dir=args.out_dir)
