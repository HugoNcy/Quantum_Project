"""Train the generator under each noise profile of the grid (plan, phase 2)."""

import argparse
import json
from pathlib import Path

import torch

from data import DATASETS
from losses import mmd_loss
from model import PhotonicGenerator
from noise import PROFILE_ORDER, PROFILE_PARAMS, expected_dim, is_lossy, make_noise

N_FLOOR_PAIRS = 16


def mmd_floor_stats(target_pool: torch.Tensor, batch_size: int) -> tuple[float, float]:
    """Mean and std of the real vs real MMD over disjoint batch pairs.

    A single pair varies by more than a factor 2 between draws (review F4),
    so every reported floor is averaged over N_FLOOR_PAIRS disjoint pairs.
    """
    values = []
    for k in range(N_FLOOR_PAIRS):
        a = target_pool[2 * k * batch_size:(2 * k + 1) * batch_size]
        b = target_pool[(2 * k + 1) * batch_size:(2 * k + 2) * batch_size]
        values.append(mmd_loss(a, b).item())
    t = torch.tensor(values)
    return t.mean().item(), t.std().item()


def train_generator(
    model: torch.nn.Module,
    target_pool: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    out_dir: str,
    meta: dict,
) -> dict:
    """Generic MMD training loop shared by the phase 2 and phase 3 scripts.

    The caller seeds torch and builds the model; this function trains it,
    saves weights and writes a JSON log carrying the metadata needed to
    rebuild the exact configuration (noise profile included, review F3).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    floor_mean, floor_std = mmd_floor_stats(target_pool, batch_size)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    # Incremental CSV so long background runs can be monitored from files.
    with open(out / "history.csv", "w") as csv:
        csv.write("step,mmd\n")
        for step in range(steps):
            # Fresh real batch from the pool, fresh latent batch inside forward.
            idx = torch.randint(0, len(target_pool), (batch_size,))
            real = target_pool[idx]
            fake = model(batch_size)
            loss = mmd_loss(fake, real)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())
            csv.write(f"{step},{loss.item():.8f}\n")
            if step % 25 == 0:
                csv.flush()

    torch.save(model.state_dict(), out / "model.pt")
    log = dict(meta)
    log.update({
        "steps": steps, "batch_size": batch_size, "lr": lr,
        "mmd_floor_mean": floor_mean, "mmd_floor_std": floor_std,
        "history": history,
    })
    (out / "log.json").write_text(json.dumps(log))
    return log


def run_grid(
    profiles: list[str],
    dataset: str = "two_gaussians",
    steps: int = 800,
    batch_size: int = 256,
    lr: float = 5e-3,
    latent_dim: int = 6,
    n_modes: int = 6,
    n_photons: int = 3,
    seed: int = 0,
    out_root: str = "runs/noise_grid",
) -> None:
    target_fn = DATASETS[dataset]
    # One pool for training batches and floor estimation, size chosen so the
    # 16 floor pairs (2*16 batches) fit disjointly.
    target_pool = target_fn(20000, seed=seed)

    for profile in profiles:
        # Reseed per profile: every run is reproducible in isolation.
        torch.manual_seed(seed)
        model = PhotonicGenerator(
            latent_dim=latent_dim, n_modes=n_modes, n_photons=n_photons,
            noise=make_noise(profile),
        )
        # FOCK guard (review F3): the profile determines the dimension, and
        # a wrong dimension here means a wrongly built layer.
        dim = model.quantum(torch.zeros(1, n_modes)).shape[1]
        assert dim == expected_dim(n_modes, n_photons, is_lossy(profile)), (
            f"{profile}: unexpected output dim {dim}"
        )
        meta = {
            "model": "photonic",
            "dataset": dataset,
            "profile": profile,
            "profile_params": PROFILE_PARAMS[profile],
            "latent_dim": latent_dim, "n_modes": n_modes, "n_photons": n_photons,
            "seed": seed,
            "fock_dim": dim,
            # Stored as text: the attribute may be a BasicState, which does
            # not serialize to JSON, and route 1 layers may not expose it.
            "input_state": str(getattr(model.quantum, "input_state", "default")),
        }
        train_generator(
            model, target_pool, steps=steps, batch_size=batch_size, lr=lr,
            out_dir=f"{out_root}/{profile}", meta=meta,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", default=PROFILE_ORDER,
                        choices=PROFILE_ORDER)
    parser.add_argument("--dataset", default="two_gaussians", choices=list(DATASETS))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default="runs/noise_grid")
    args = parser.parse_args()
    run_grid(args.profiles, dataset=args.dataset, steps=args.steps,
             seed=args.seed, out_root=args.out_root)
