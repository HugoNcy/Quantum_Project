"""Checks for the cross-profile weight transfer machinery (phase 2 gate).

Five checks, run before any grid training:
1. output_keys contract: a clean layer exposes exactly the C(8,3) = 56
   occupation tuples with 3 photons, a lossy (P4) layer exposes the 84
   tuples with 0 to 3 photons, and the clean set is contained in the
   lossy set (the transfer maps columns through these keys).
2. Mechanical alignment: after transfer_generator from P0 to P4, every
   column of the destination adapter equals the source column of the
   same occupation tuple, and the 28 loss-only states get zero columns.
3. Semantic ordering: a structurally lossy but physically near-clean
   destination (transmittance 0.9999) must reproduce the source outputs
   on identical latents after transfer. This fails grossly if column j
   of the layer output did not correspond to output_keys[j].
4. MeshGenerator smoke: both mesh types build under clean and lossy
   profiles with the expected output dimensions, train for two steps,
   and support the same transfer path used by mismatch_matrix.
5. Pipeline: a two-step training run written by train_generator can be
   rebuilt from its log by build_generator_from_log and evaluated.

Runnable with pytest or directly; direct runs log measured values to
tests/test_transfer_results.json.
"""

import json
import sys
from pathlib import Path

import perceval as pcvl
import torch

# The src modules import each other by bare name (script style), so the
# tests put src itself on the path instead of importing the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import two_gaussians
from losses import mmd_loss
from mismatch_matrix import build_generator_from_log, eval_mmd
from model import MeshGenerator, PhotonicGenerator
from noise import make_noise, transfer_generator
from train_noise_grid import train_generator

SEED = 0
N_MODES = 6
N_PHOTONS = 3
DIM_CLEAN = 56
DIM_LOSSY = 84


def _keys(gen) -> list[tuple]:
    return [tuple(int(v) for v in k) for k in gen.quantum.output_keys]


def test_output_keys_contract():
    torch.manual_seed(SEED)
    clean = PhotonicGenerator(noise=None)
    lossy = PhotonicGenerator(noise=make_noise("P4"))
    ck, lk = _keys(clean), _keys(lossy)
    assert len(ck) == DIM_CLEAN, f"clean key count {len(ck)}"
    assert len(lk) == DIM_LOSSY, f"lossy key count {len(lk)}"
    assert all(sum(k) == N_PHOTONS for k in ck), "clean keys must have 3 photons"
    assert all(0 <= sum(k) <= N_PHOTONS for k in lk), "lossy keys out of range"
    common = set(ck) & set(lk)
    assert len(common) == DIM_CLEAN, f"only {len(common)} shared states"
    # The layer output width must match the key count on both layers.
    assert clean.quantum(torch.zeros(1, N_MODES)).shape[1] == DIM_CLEAN
    assert lossy.quantum(torch.zeros(1, N_MODES)).shape[1] == DIM_LOSSY
    return {"n_clean": len(ck), "n_lossy": len(lk), "n_common": len(common)}


def test_transfer_column_alignment():
    torch.manual_seed(SEED)
    src = PhotonicGenerator(noise=None)
    dst = PhotonicGenerator(noise=make_noise("P4"))
    transfer_generator(src, dst)
    src_index = {k: i for i, k in enumerate(_keys(src))}
    dst_w = dst.adapter[0].weight
    src_w = src.adapter[0].weight
    n_zero = 0
    for j, key in enumerate(_keys(dst)):
        col = dst_w[:, j]
        if key in src_index:
            assert torch.equal(col, src_w[:, src_index[key]]), f"column {j} mismatch"
        else:
            assert torch.all(col == 0), f"loss-only column {j} not zeroed"
            n_zero += 1
    assert n_zero == DIM_LOSSY - DIM_CLEAN, f"{n_zero} zero columns"
    # Bias and the adapter tail share shapes and must have been copied.
    assert torch.equal(dst.adapter[0].bias, src.adapter[0].bias)
    assert torch.equal(dst.adapter[2].weight, src.adapter[2].weight)
    return {"n_zero_columns": n_zero}


def test_transfer_preserves_near_clean_outputs():
    torch.manual_seed(SEED)
    src = PhotonicGenerator(noise=None)
    # Lossy structure (84 states) but physically almost clean, so the
    # transferred generator must behave like the source.
    dst = PhotonicGenerator(noise=pcvl.NoiseModel(transmittance=0.9999))
    assert dst.quantum(torch.zeros(1, N_MODES)).shape[1] == DIM_LOSSY
    transfer_generator(src, dst)
    g1 = torch.Generator().manual_seed(SEED + 1)
    g2 = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        out_src = src(64, generator=g1)
        out_dst = dst(64, generator=g2)
    err = float((out_src - out_dst).abs().max())
    assert err < 1e-2, f"transferred outputs diverge: {err}"
    return err


def test_mesh_generator_smoke():
    target = two_gaussians(512, seed=SEED)
    dims = {}
    for mesh_type, profile, expected in (
        ("mzi", "P0", DIM_CLEAN),
        ("tritter", "P4", DIM_LOSSY),
    ):
        torch.manual_seed(SEED)
        gen = MeshGenerator(mesh_type=mesh_type, noise=make_noise(profile))
        dim = gen.quantum(torch.zeros(1, N_MODES)).shape[1]
        assert dim == expected, f"{mesh_type}/{profile}: dim {dim}"
        # Two optimizer steps: the loss must stay finite and grads flow.
        opt = torch.optim.Adam(gen.parameters(), lr=5e-3)
        for _ in range(2):
            loss = mmd_loss(gen(64), target[:64])
            opt.zero_grad()
            loss.backward()
            opt.step()
            assert torch.isfinite(loss), f"{mesh_type}/{profile}: loss {loss}"
        dims[f"{mesh_type}_{profile}"] = dim
    # Same transfer path as mismatch_matrix: tritter P0 to tritter P4.
    torch.manual_seed(SEED)
    src = MeshGenerator(mesh_type="tritter", noise=None)
    torch.manual_seed(SEED)
    dst = MeshGenerator(mesh_type="tritter", noise=make_noise("P4"))
    transfer_generator(src, dst)
    with torch.no_grad():
        out = dst(32)
    assert torch.isfinite(out).all(), "transferred mesh generator output not finite"
    return dims


def test_train_and_rebuild_pipeline():
    torch.manual_seed(SEED)
    pool = two_gaussians(4096, seed=SEED)
    model = PhotonicGenerator(noise=make_noise("P0"))
    meta = {
        "model": "photonic", "dataset": "two_gaussians", "profile": "P0",
        "latent_dim": 6, "n_modes": N_MODES, "n_photons": N_PHOTONS,
        "seed": SEED,
    }
    out_dir = Path(__file__).resolve().parents[1] / "runs" / "smoke" / "P0"
    log = train_generator(model, pool, steps=2, batch_size=64, lr=5e-3,
                          out_dir=str(out_dir), meta=meta)
    assert all(v == v for v in log["history"]), "NaN in training history"
    rebuilt = build_generator_from_log(log, "P0")
    rebuilt.load_state_dict(torch.load(out_dir / "model.pt", weights_only=True))
    eval_pool = two_gaussians(8 * 256, seed=SEED + 1)
    mean, std = eval_mmd(rebuilt, eval_pool, eval_seed=SEED + 100)
    assert mean == mean and std == std, "eval MMD is NaN"
    return {"history": log["history"], "eval_mmd_mean": mean, "eval_mmd_std": std}


if __name__ == "__main__":
    results = {
        "seed": SEED,
        "n_modes": N_MODES,
        "n_photons": N_PHOTONS,
        "output_keys": test_output_keys_contract(),
        "column_alignment": test_transfer_column_alignment(),
        "near_clean_transfer_max_err": test_transfer_preserves_near_clean_outputs(),
        "mesh_smoke_dims": test_mesh_generator_smoke(),
        "pipeline": test_train_and_rebuild_pipeline(),
    }
    out_path = Path(__file__).with_name("test_transfer_results.json")
    out_path.write_text(json.dumps(results, indent=2))
