"""Checks derived from the adversarial review (docs/REVIEW.md).

Four checks:
1. F1: the effective parameter budget (Jacobian rank of the output
   probabilities with respect to the trainable phases) is comparable
   between the mzi and tritter circuits, not only the nominal name count.
2. F8: input sensitivity at the contract level: max per-dimension std
   above 1e-2 AND mean per-dimension std above 1e-3, so the sensitivity
   is distributed and not carried by a single output component.
3. F7: two successive tritter() decompositions are elementwise equal
   (the solver has no exposed seed, so this pins the observed determinism).
4. F9: the "x" parameters appear in the circuit in mode order, so the
   insertion-order mapping used by MerLin matches the names.

Runnable with pytest or directly; direct runs log measured values to
tests/test_review_findings_results.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import merlin
from src.circuits import build_circuit, tritter

SEED = 0
N_MODES = 6
N_PHOTONS = 3
FOCK = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)


def _make_layer(mesh_type: str) -> merlin.QuantumLayer:
    # Route 2 construction: raw pcvl.Circuit plus parameter prefixes.
    return merlin.QuantumLayer(
        input_size=N_MODES,
        circuit=build_circuit(mesh_type, N_MODES),
        n_photons=N_PHOTONS,
        input_parameters=["x"],
        trainable_parameters=["theta"],
        measurement_strategy=FOCK,
    )


def _jacobian_rank(layer: merlin.QuantumLayer, n_inputs: int = 4, rtol: float = 1e-7):
    """Rank of d(output probabilities) / d(trainable phases).

    Rows are all output components stacked over n_inputs fixed random
    inputs; columns are all trainable scalar parameters. Directions with
    singular value below rtol * s_max are counted as flat.
    """
    g = torch.Generator().manual_seed(SEED)
    x = torch.rand(n_inputs, N_MODES, generator=g) * torch.pi
    params = [p for _, p in sorted(layer.named_parameters())]
    n_params = sum(p.numel() for p in params)
    out = layer(x)
    rows = []
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            grads = torch.autograd.grad(
                out[i, j], params, retain_graph=True, allow_unused=True
            )
            row = torch.cat(
                [
                    (gr if gr is not None else torch.zeros_like(p)).reshape(-1)
                    for gr, p in zip(grads, params)
                ]
            )
            rows.append(row)
    jac = torch.stack(rows).double()
    sv = torch.linalg.svdvals(jac)
    rank = int((sv > rtol * sv[0]).sum().item())
    return rank, n_params


def test_effective_budget_parity():
    torch.manual_seed(SEED)
    ranks = {}
    for mesh_type in ("mzi", "tritter"):
        rank, n_params = _jacobian_rank(_make_layer(mesh_type))
        ranks[mesh_type] = {"rank": rank, "nominal": n_params}
    r_mzi = ranks["mzi"]["rank"]
    r_tri = ranks["tritter"]["rank"]
    assert abs(r_tri - r_mzi) / r_mzi <= 0.10, f"effective budgets differ: {ranks}"
    return ranks


def test_input_sensitivity_strict():
    torch.manual_seed(SEED)
    stds = {}
    for mesh_type in ("mzi", "tritter"):
        layer = _make_layer(mesh_type)
        x = torch.rand(16, N_MODES) * torch.pi
        out = layer(x)
        per_dim_std = out.std(dim=0)
        max_std = float(per_dim_std.max())
        mean_std = float(per_dim_std.mean())
        assert max_std > 1e-2, f"{mesh_type}: max std {max_std} below contract"
        assert mean_std > 1e-3, f"{mesh_type}: sensitivity concentrated, mean std {mean_std}"
        stds[mesh_type] = {"max_std": max_std, "mean_std": mean_std}
    return stds


def test_tritter_decomposition_deterministic():
    u1 = np.array(tritter().compute_unitary())
    u2 = np.array(tritter().compute_unitary())
    err = float(np.abs(u1 - u2).max())
    assert err < 1e-8, f"tritter decomposition drifted between calls: {err}"
    return err


def test_x_parameters_in_mode_order():
    for mesh_type in ("mzi", "tritter"):
        params = build_circuit(mesh_type, N_MODES).get_parameters()
        x_names = [p.name for p in params if p.name.startswith("x")]
        assert x_names == [f"x{i}" for i in range(N_MODES)], x_names
    return True


if __name__ == "__main__":
    results = {
        "seed": SEED,
        "n_modes": N_MODES,
        "n_photons": N_PHOTONS,
        "effective_budget": test_effective_budget_parity(),
        "sensitivity": test_input_sensitivity_strict(),
        "tritter_determinism_max_err": test_tritter_decomposition_deterministic(),
        "x_order_ok": test_x_parameters_in_mode_order(),
    }
    out_path = Path(__file__).with_name("test_review_findings_results.json")
    out_path.write_text(json.dumps(results, indent=2))
