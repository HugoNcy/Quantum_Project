"""Phase 3: MZI vs tritter comparison across the noise grid."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import DATASETS
from mismatch_matrix import mismatch_matrix
from model import MeshGenerator
from noise import PROFILE_ORDER, PROFILE_PARAMS, expected_dim, is_lossy, make_noise
from train_noise_grid import train_generator

MESH_TYPES = ("mzi", "tritter")


def run_comparison(
    mesh_types: list[str],
    profiles: list[str],
    dataset: str = "two_gaussians",
    steps: int = 800,
    batch_size: int = 256,
    lr: float = 5e-3,
    latent_dim: int = 6,
    n_modes: int = 6,
    n_photons: int = 3,
    seed: int = 0,
    out_root: str = "runs/mesh_compare",
    fig_dir: str = "figures",
) -> None:
    target_pool = DATASETS[dataset](20000, seed=seed)

    for mesh_type in mesh_types:
        for profile in profiles:
            # Every (mesh, profile) run is reseeded and self-contained.
            torch.manual_seed(seed)
            model = MeshGenerator(
                mesh_type=mesh_type, latent_dim=latent_dim, n_modes=n_modes,
                n_photons=n_photons, noise=make_noise(profile),
            )
            # FOCK guard: dimension must follow the profile, nothing else.
            dim = model.quantum(torch.zeros(1, n_modes)).shape[1]
            assert dim == expected_dim(n_modes, n_photons, is_lossy(profile)), (
                f"{mesh_type}/{profile}: unexpected output dim {dim}"
            )
            meta = {
                "model": f"mesh:{mesh_type}",
                "dataset": dataset,
                "profile": profile,
                "profile_params": PROFILE_PARAMS[profile],
                "latent_dim": latent_dim, "n_modes": n_modes, "n_photons": n_photons,
                "seed": seed,
                "fock_dim": dim,
                "input_state": str(getattr(model.quantum, "input_state", "default")),
                # Trainable phase count logged per run so the fair-budget
                # requirement of the comparison is auditable from the logs.
                "n_thetas": sum(
                    p.numel() for n, p in model.quantum.named_parameters()
                ),
            }
            train_generator(
                model, target_pool, steps=steps, batch_size=batch_size, lr=lr,
                out_dir=f"{out_root}/{mesh_type}/{profile}", meta=meta,
            )


def final_table(
    mesh_types: list[str],
    profiles: list[str],
    out_root: str = "runs/mesh_compare",
    fig_dir: str = "figures",
) -> dict:
    """Mismatch matrix per mesh, then the final mesh x profile table.

    The matched-profile (diagonal) MMD of each mesh's mismatch matrix is
    the headline number: final generative quality when the deployment
    noise equals the training noise.
    """
    results = {}
    for mesh_type in mesh_types:
        results[mesh_type] = mismatch_matrix(
            runs_root=f"{out_root}/{mesh_type}", profiles=profiles,
            label=f"mesh_{mesh_type}", fig_dir=fig_dir,
        )

    figs = Path(fig_dir)
    figs.mkdir(parents=True, exist_ok=True)

    # Long-format CSV: one row per (mesh, profile) with mean and std.
    lines = ["mesh,profile,mmd_mean,mmd_std,floor_mean"]
    for mesh_type in mesh_types:
        r = results[mesh_type]
        for p in profiles:
            cell = r["matrix"][p][p]
            lines.append(
                f"{mesh_type},{p},{cell['mmd_mean']:.6f},{cell['mmd_std']:.6f},"
                f"{r['floor_mean']:.6f}"
            )
    (figs / "final_table.csv").write_text("\n".join(lines) + "\n")

    # Ranking stability figure: matched-profile MMD per profile, one line
    # per mesh. If the lines never cross, the ranking survives the noise.
    fig, ax = plt.subplots(figsize=(6, 4))
    for mesh_type in mesh_types:
        r = results[mesh_type]
        diag = [r["matrix"][p][p]["mmd_mean"] for p in profiles]
        err = [r["matrix"][p][p]["mmd_std"] for p in profiles]
        ax.errorbar(range(len(profiles)), diag, yerr=err, fmt="o-", capsize=3,
                    label=mesh_type)
    # Both meshes share the evaluation pool, so one floor line suffices.
    ax.axhline(results[mesh_types[0]]["floor_mean"], color="gray",
               linestyle="--", label="real vs real floor")
    ax.set_xticks(range(len(profiles)), profiles)
    ax.set_xlabel("noise profile (train = eval)")
    ax.set_ylabel("final MMD^2")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "ranking_vs_noise.png", dpi=150)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-types", nargs="+", default=list(MESH_TYPES),
                        choices=MESH_TYPES)
    parser.add_argument("--profiles", nargs="+", default=PROFILE_ORDER,
                        choices=PROFILE_ORDER)
    parser.add_argument("--dataset", default="two_gaussians", choices=list(DATASETS))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default="runs/mesh_compare")
    parser.add_argument("--skip-train", action="store_true",
                        help="only rebuild the table and figures from existing runs")
    args = parser.parse_args()
    if not args.skip_train:
        run_comparison(args.mesh_types, args.profiles, dataset=args.dataset,
                       steps=args.steps, seed=args.seed, out_root=args.out_root)
    final_table(args.mesh_types, args.profiles, out_root=args.out_root)
