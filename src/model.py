"""Hybrid photonic generator: latent -> quantum layer -> classical adapter."""

import perceval as pcvl
import torch
import torch.nn as nn

import merlin


def build_quantum_layer(
    n_modes: int = 6,
    n_photons: int = 3,
    noise: pcvl.NoiseModel | None = None,
) -> merlin.QuantumLayer:
    """Angle-encoded MZI mesh layer with the FOCK space forced.

    ComputationSpace.FOCK is mandatory on every layer of the project so that
    clean and noisy outputs always live in the same basis (see plan, trap 2).
    """
    # Sandwich structure: mesh, encoding, mesh. Encoding phases applied
    # directly on the input Fock basis state would only add a global phase
    # and leave output probabilities unchanged, so mixing must come first.
    builder = merlin.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(model="mzi", trainable=True)
    builder.add_angle_encoding(name="x")
    builder.add_entangling_layer(model="mzi", trainable=True)
    strategy = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)
    return merlin.QuantumLayer(
        input_size=n_modes,
        builder=builder,
        n_photons=n_photons,
        measurement_strategy=strategy,
        noise=noise,
    )


class PhotonicGenerator(nn.Module):
    """z ~ N(0, I) -> linear map to angles -> QuantumLayer -> linear adapter -> R^2."""

    def __init__(self, latent_dim: int = 6, n_modes: int = 6, n_photons: int = 3,
                 out_dim: int = 2, noise: pcvl.NoiseModel | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(latent_dim, n_modes)
        self.quantum = build_quantum_layer(n_modes, n_photons, noise)
        fock_dim = self.quantum(torch.zeros(1, n_modes)).shape[1]
        self.adapter = nn.Sequential(
            nn.Linear(fock_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim, generator=generator)
        angles = torch.pi * torch.tanh(self.encoder(z))
        probs = self.quantum(angles)
        return self.adapter(probs)


def copy_circuit_params(src: merlin.QuantumLayer, dst: merlin.QuantumLayer) -> None:
    """Copy trainable circuit phases by parameter name.

    Full state_dict copy breaks when noise models differ in structure
    (transmittance adds loss modes, see plan, trap 3), so we match names.
    """
    src_params = dict(src.named_parameters())
    with torch.no_grad():
        for name, p in dst.named_parameters():
            if name in src_params:
                p.copy_(src_params[name])
