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
    # First trainable Clements mesh of MZIs, mixes the modes before encoding.
    builder.add_entangling_layer(model="mzi", trainable=True)
    # One input phase shifter per mode, named with prefix "x" (the naming
    # contract with the training loop; these phases carry the data).
    builder.add_angle_encoding(name="x")
    # Second trainable mesh after the encoding, completes the sandwich.
    builder.add_entangling_layer(model="mzi", trainable=True)
    # Force the full Fock space so the output dimension does not silently
    # change when a noise model is attached (project trap 2).
    strategy = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)
    return merlin.QuantumLayer(
        # Route 1: pass the builder itself, never builder.build() (trap 1).
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
        # Classical map from the latent vector to one angle per mode.
        self.encoder = nn.Linear(latent_dim, n_modes)
        # Differentiable photonic layer (probabilities over Fock states).
        self.quantum = build_quantum_layer(n_modes, n_photons, noise)
        # One dummy forward to discover the output dimension, which depends
        # on n_modes, n_photons and the noise model (losses extend the space).
        fock_dim = self.quantum(torch.zeros(1, n_modes)).shape[1]
        # Classical head mapping the Fock distribution to a 2D sample.
        self.adapter = nn.Sequential(
            nn.Linear(fock_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        # Classical gaussian latent, the only source of randomness.
        z = torch.randn(batch_size, self.latent_dim, generator=generator)
        # tanh bounds the encoder output, scaled to angles in (-pi, pi).
        angles = torch.pi * torch.tanh(self.encoder(z))
        # Exact differentiable output distribution over Fock states.
        probs = self.quantum(angles)
        # Project the distribution to a point in R^out_dim.
        return self.adapter(probs)


def build_mesh_layer(
    mesh_type: str,
    n_modes: int = 6,
    n_photons: int = 3,
    noise: pcvl.NoiseModel | None = None,
    input_state: list[int] | None = None,
) -> merlin.QuantumLayer:
    """Route 2 layer over a raw pcvl.Circuit from circuits.build_circuit.

    Used for the phase 3 mesh comparison: both mesh types go through the
    exact same construction path, with the parameter-prefix contract
    ("x" for inputs, "theta" for trainables) declared manually. The input
    state is explicit (review F10) so both meshes receive the photons at
    the same positions regardless of MerLin defaults.
    """
    from circuits import build_circuit

    if input_state is None:
        # One photon on every other mode, e.g. [1, 0, 1, 0, 1, 0] at 6/3.
        input_state = [1 if i % 2 == 0 and i // 2 < n_photons else 0
                       for i in range(n_modes)]
    # Same non-negotiable FOCK forcing as every other layer of the project.
    strategy = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)
    return merlin.QuantumLayer(
        input_size=n_modes,
        circuit=build_circuit(mesh_type, n_modes),
        input_state=input_state,
        input_parameters=["x"],
        trainable_parameters=["theta"],
        measurement_strategy=strategy,
        noise=noise,
    )


class MeshGenerator(nn.Module):
    """Same pipeline as PhotonicGenerator but over an explicit mesh circuit.

    latent z -> linear encoder -> angles -> route 2 QuantumLayer (mzi or
    tritter sandwich from circuits.py) -> linear adapter -> R^out_dim.
    """

    def __init__(self, mesh_type: str, latent_dim: int = 6, n_modes: int = 6,
                 n_photons: int = 3, out_dim: int = 2,
                 noise: pcvl.NoiseModel | None = None,
                 input_state: list[int] | None = None):
        super().__init__()
        self.mesh_type = mesh_type
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(latent_dim, n_modes)
        self.quantum = build_mesh_layer(mesh_type, n_modes, n_photons, noise, input_state)
        # Output dimension depends on the noise model (losses extend the
        # space), so it is read from a dummy forward like PhotonicGenerator.
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
    Note: unlike the canonical version in the plan, this one does not check
    shapes before copying; a same-named parameter with a different shape
    would raise instead of being skipped.
    """
    # Name -> tensor view of every trainable parameter of the source layer.
    src_params = dict(src.named_parameters())
    # no_grad because this is a raw weight transfer, not a training step.
    with torch.no_grad():
        for name, p in dst.named_parameters():
            # Only parameters present in both layers are copied; keys that
            # exist on one side only (noise internals) are left untouched.
            if name in src_params:
                p.copy_(src_params[name])
