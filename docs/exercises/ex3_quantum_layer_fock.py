"""Exercise 3 -- QuantumLayer and the Fock space (THE critical project rule).

Goal: instantiate a QuantumLayer (4 modes, 2 photons) and see with your own
eyes why the project forces ComputationSpace.FOCK everywhere: the default
(no-bunching) output has dimension C(4,2)=6, the full Fock space C(5,2)=10.
Under noise MerLin silently switches to the full space -- so we always force it.
"""

import torch
import merlin

def make_builder():
    builder = merlin.CircuitBuilder(n_modes=4)
    builder.add_angle_encoding(name="x")
    builder.add_entangling_layer(model="mzi", trainable=True)
    return builder

x = torch.rand(8, 4) * torch.pi

# --- Layer A: default measurement (no-bunching subspace) ---
layer_default = merlin.QuantumLayer(
    input_size=4,
    builder=make_builder(),
    n_photons=2,
)
out_default = layer_default(x)
print("default output shape:", tuple(out_default.shape), "-> C(4,2) = 6")

# --- Layer B: full Fock space, forced (PROJECT RULE, non-negotiable) ---
strategy = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)
layer_fock = merlin.QuantumLayer(
    input_size=4,
    builder=make_builder(),
    n_photons=2,
    measurement_strategy=strategy,
)
out_fock = layer_fock(x)
print("FOCK output shape:", tuple(out_fock.shape), "-> C(5,2) = 10")

# Both are probability distributions: rows sum to 1.
print("rows sum to 1 (default):", torch.allclose(out_default.sum(dim=1), torch.ones(8)))
print("rows sum to 1 (FOCK):   ", torch.allclose(out_fock.sum(dim=1), torch.ones(8)))
