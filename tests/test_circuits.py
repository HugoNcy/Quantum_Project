"""Validation checks for src/circuits.py (plan, phase 3).

Three checks:
1. The isolated tritter unitary equals the 3x3 DFT up to a global phase.
2. Both circuits, wrapped in a QuantumLayer with the FOCK space forced,
   pass the input sensitivity test (output std over a random input batch
   of order 1e-2, not 1e-8).
3. The number of trainable parameters is comparable between mzi and
   tritter at equal n_modes.

Runnable with pytest or directly; direct runs log measured values to
tests/test_circuits_results.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import merlin
from src.circuits import build_circuit, dft3_matrix, tritter

SEED = 0
N_MODES = 6
N_PHOTONS = 3


def test_tritter_is_dft3_up_to_global_phase():
    u = np.array(tritter().compute_unitary())
    d = np.array(dft3_matrix())
    ratio = u @ d.conj().T
    phase = ratio[0, 0] / abs(ratio[0, 0])
    err = float(np.abs(u - phase * d).max())
    assert err < 1e-6
    return err


def test_input_sensitivity():
    torch.manual_seed(SEED)
    strategy = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)
    stds = {}
    for mesh_type in ("mzi", "tritter"):
        layer = merlin.QuantumLayer(
            input_size=N_MODES,
            circuit=build_circuit(mesh_type, N_MODES),
            n_photons=N_PHOTONS,
            input_parameters=["x"],
            trainable_parameters=["theta"],
            measurement_strategy=strategy,
        )
        x = torch.rand(16, N_MODES) * torch.pi
        out = layer(x)
        std = float(out.std(dim=0).max())
        assert std > 1e-3, f"{mesh_type}: input-insensitive circuit, std={std}"
        stds[mesh_type] = std
    return stds


def test_trainable_count_comparable():
    counts = {}
    for mesh_type in ("mzi", "tritter"):
        params = build_circuit(mesh_type, N_MODES).get_parameters()
        names = [p.name for p in params]
        assert all(n.startswith(("x", "theta")) for n in names)
        assert sum(n.startswith("x") for n in names) == N_MODES
        counts[mesh_type] = sum(n.startswith("theta") for n in names)
    ratio = counts["tritter"] / counts["mzi"]
    assert 0.8 <= ratio <= 1.25, f"unbalanced budgets: {counts}"
    return counts


if __name__ == "__main__":
    results = {
        "seed": SEED,
        "n_modes": N_MODES,
        "n_photons": N_PHOTONS,
        "tritter_dft_max_error": test_tritter_is_dft3_up_to_global_phase(),
        "max_output_std": test_input_sensitivity(),
        "trainable_counts": test_trainable_count_comparable(),
    }
    out_path = Path(__file__).with_name("test_circuits_results.json")
    out_path.write_text(json.dumps(results, indent=2))
