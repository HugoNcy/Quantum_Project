"""Noise mismatch matrix: train under profile A, evaluate under profile B."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import DATASETS
from losses import mmd_loss
from model import MeshGenerator, PhotonicGenerator
from noise import PROFILE_ORDER, PROFILE_PARAMS, make_noise, transfer_generator
from train_noise_grid import mmd_floor_stats

EVAL_BATCH = 256
N_EVAL_BATCHES = 8


def build_generator_from_log(log: dict, profile: str) -> torch.nn.Module:
    """Rebuild the generator described by a run log under a given profile.

    The log's "model" field selects the class ("photonic" for the builder
    based MZI generator, "mesh:mzi" / "mesh:tritter" for the raw-circuit
    generators of phase 3), so this script works for both phases.
    """
    kind = log["model"]
    noise = make_noise(profile)
    if kind == "photonic":
        return PhotonicGenerator(
            latent_dim=log["latent_dim"], n_modes=log["n_modes"],
            n_photons=log["n_photons"], noise=noise,
        )
    if kind.startswith("mesh:"):
        return MeshGenerator(
            mesh_type=kind.split(":", 1)[1], latent_dim=log["latent_dim"],
            n_modes=log["n_modes"], n_photons=log["n_photons"], noise=noise,
        )
    raise ValueError(f"unknown model kind in log: {kind}")


def eval_mmd(model: torch.nn.Module, eval_pool: torch.Tensor, eval_seed: int) -> tuple[float, float]:
    """Mean and std of the MMD over fixed generated vs real batch pairs."""
    model.eval()
    values = []
    with torch.no_grad():
        for r in range(N_EVAL_BATCHES):
            # Deterministic latents per repeat, disjoint real slices.
            gen = torch.Generator().manual_seed(eval_seed + r)
            fake = model(EVAL_BATCH, generator=gen)
            real = eval_pool[r * EVAL_BATCH:(r + 1) * EVAL_BATCH]
            values.append(mmd_loss(fake, real).item())
    t = torch.tensor(values)
    return t.mean().item(), t.std().item()


def mismatch_matrix(
    runs_root: str,
    profiles: list[str],
    label: str,
    fig_dir: str = "figures",
) -> dict:
    root = Path(runs_root)
    # Load every training run of the grid once.
    logs = {}
    for profile in profiles:
        logs[profile] = json.loads((root / profile / "log.json").read_text())

    # All grid runs share dataset and seed, so any log works as reference.
    ref = logs[profiles[0]]
    # Evaluation data: same target, seed distinct from training batches.
    # Sized for the larger consumer: the floor uses 2 * N_FLOOR_PAIRS
    # disjoint batches, the matrix cells use N_EVAL_BATCHES slices.
    eval_pool = DATASETS[ref["dataset"]](
        max(2 * 16 * EVAL_BATCH, N_EVAL_BATCHES * EVAL_BATCH), seed=ref["seed"] + 1
    )
    floor_mean, floor_std = mmd_floor_stats(eval_pool, EVAL_BATCH)

    matrix = {}
    for a in profiles:
        # Source generator: built under its own profile, weights loaded
        # directly (same structure, so a plain state_dict load is safe).
        torch.manual_seed(ref["seed"])
        src = build_generator_from_log(logs[a], a)
        src.load_state_dict(
            torch.load(root / a / "model.pt", weights_only=True)
        )
        matrix[a] = {}
        for b in profiles:
            # Destination generator: architecture and physics under B,
            # trained weights transferred from A (never state_dict across
            # profiles: lossy layers have a different structure).
            torch.manual_seed(ref["seed"])
            dst = build_generator_from_log(logs[a], b)
            transfer_generator(src, dst)
            mean, std = eval_mmd(dst, eval_pool, eval_seed=ref["seed"] + 100)
            matrix[a][b] = {"mmd_mean": mean, "mmd_std": std}

    # Diagonal must beat the off-diagonal on average, else suspect a bug
    # (plan, phase 2 acceptance criterion).
    diag = [matrix[p][p]["mmd_mean"] for p in profiles]
    off = [matrix[a][b]["mmd_mean"] for a in profiles for b in profiles if a != b]
    results = {
        "label": label,
        "dataset": ref["dataset"],
        "profiles": profiles,
        "profile_params": {p: PROFILE_PARAMS[p] for p in profiles},
        "eval_batch": EVAL_BATCH,
        "n_eval_batches": N_EVAL_BATCHES,
        "floor_mean": floor_mean,
        "floor_std": floor_std,
        "matrix": matrix,
        "diag_mean": sum(diag) / len(diag),
        "offdiag_mean": sum(off) / len(off),
        "suspect_bug": sum(diag) / len(diag) > sum(off) / len(off),
    }
    (root / f"mismatch_{label}.json").write_text(json.dumps(results, indent=2))

    # CSV of the mean values, rows = training profile, cols = eval profile.
    lines = ["train\\eval," + ",".join(profiles)]
    for a in profiles:
        lines.append(a + "," + ",".join(f"{matrix[a][b]['mmd_mean']:.6f}" for b in profiles))
    (root / f"mismatch_{label}.csv").write_text("\n".join(lines) + "\n")

    figs = Path(fig_dir)
    figs.mkdir(parents=True, exist_ok=True)

    # Heatmap of the K x K matrix.
    values = torch.tensor([[matrix[a][b]["mmd_mean"] for b in profiles] for a in profiles])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, cmap="viridis")
    ax.set_xticks(range(len(profiles)), profiles)
    ax.set_yticks(range(len(profiles)), profiles)
    ax.set_xlabel("evaluation profile")
    ax.set_ylabel("training profile")
    for i in range(len(profiles)):
        for j in range(len(profiles)):
            ax.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center",
                    color="white" if values[i, j] < values.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="MMD^2 vs target")
    ax.set_title(f"noise mismatch, {label}, {ref['dataset']}")
    fig.tight_layout()
    fig.savefig(figs / f"mismatch_heatmap_{label}.png", dpi=150)

    # Final matched-profile MMD against the indistinguishability level.
    indist = [PROFILE_PARAMS[p].get("indistinguishability", 1.0) for p in profiles]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(indist, diag, "o-", label="train = eval profile")
    ax.axhline(floor_mean, color="gray", linestyle="--", label="real vs real floor")
    for x, y, p in zip(indist, diag, profiles):
        ax.annotate(p, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("indistinguishability")
    ax.set_ylabel("final MMD^2")
    # Reversed axis so noise grows from left to right along the curve.
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / f"mmd_final_vs_indistinguishability_{label}.png", dpi=150)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs/noise_grid")
    parser.add_argument("--profiles", nargs="+", default=PROFILE_ORDER,
                        choices=PROFILE_ORDER)
    parser.add_argument("--label", default="mzi")
    parser.add_argument("--fig-dir", default="figures")
    args = parser.parse_args()
    mismatch_matrix(args.runs_root, args.profiles, args.label, args.fig_dir)
