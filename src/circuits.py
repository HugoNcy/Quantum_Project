"""Pure Perceval circuits for the mesh comparison: MZI (Clements) vs tritter.

Contract with the training loop (see plan, phase 3): input encoding
parameters are named "x0", "x1", ... and trainable parameters are named
"theta0", "theta1", ... The training loop declares them by prefix and
knows nothing about the internal topology.

Both mesh types follow the mandatory sandwich structure:
mesh, angle encoding, mesh. Encoding placed before the first mesh would
only add a global phase on the input Fock state.
"""

import itertools

import numpy as np
import perceval as pcvl


def dft3_matrix() -> pcvl.Matrix:
    """Ideal tritter unitary: 3x3 DFT, coefficients are cube roots of unity."""
    w = np.exp(2j * np.pi / 3)
    u = np.array([[1, 1, 1], [1, w, w * w], [1, w * w, w]]) / np.sqrt(3)
    return pcvl.Matrix(u)


def tritter() -> pcvl.Circuit:
    """Fixed 3-mode circuit equal to the DFT 3x3 up to a global phase.

    Decomposed into beam splitters and phase shifters by Reck (triangle
    shape) on 3 modes. All phases are fixed numeric values, so none of
    them is picked up by the "x" or "theta" prefixes.
    """
    mzi = (
        pcvl.BS()
        // (0, pcvl.PS(phi=pcvl.P("phi_a")))
        // pcvl.BS()
        // (0, pcvl.PS(phi=pcvl.P("phi_b")))
    )
    return pcvl.Circuit.decomposition(
        dft3_matrix(),
        mzi,
        phase_shifter_fn=pcvl.PS,
        shape=pcvl.InterferometerShape.TRIANGLE,
    )


def _mzi_mesh(n_modes: int, counter) -> pcvl.Circuit:
    """Rectangular (Clements) mesh of MZIs with 2 trainable phases each."""

    def mzi(_: int) -> pcvl.Circuit:
        c = pcvl.Circuit(2)
        c.add(0, pcvl.BS())
        c.add(0, pcvl.PS(pcvl.P(f"theta{next(counter)}")))
        c.add(0, pcvl.BS())
        c.add(0, pcvl.PS(pcvl.P(f"theta{next(counter)}")))
        return c

    return pcvl.GenericInterferometer(
        n_modes, mzi, shape=pcvl.InterferometerShape.RECTANGLE
    )


def _tritter_mesh(n_modes: int, n_layers: int, counter) -> pcvl.Circuit:
    """Layers of tritters on shifted triplets with trainable phases between.

    Layer offsets cycle over the values that fit at least one tritter,
    e.g. triplets (0,1,2), (3,4,5) then (1,2,3), (4,5,6), then (2,3,4).
    A layer of n_modes trainable phase shifters sits between consecutive
    tritter layers. With n_layers = n_modes this gives n_modes*(n_modes-1)
    trainable phases per mesh, the same count as the MZI mesh.
    """
    offsets = [o for o in (0, 1, 2) if o + 3 <= n_modes]
    base = tritter()
    c = pcvl.Circuit(n_modes)
    for layer in range(n_layers):
        offset = offsets[layer % len(offsets)]
        for start in range(offset, n_modes - 2, 3):
            c.add(start, base.copy(), merge=True)
        if layer < n_layers - 1:
            for i in range(n_modes):
                c.add(i, pcvl.PS(pcvl.P(f"theta{next(counter)}")))
    return c


def build_circuit(mesh_type: str, n_modes: int) -> pcvl.Circuit:
    """Sandwich circuit: entangling mesh, angle encoding, entangling mesh.

    mesh_type is "mzi" or "tritter". Input parameters are "x0".."x{n-1}"
    (one phase shifter per mode), trainable parameters are "theta*".
    """
    if mesh_type not in ("mzi", "tritter"):
        raise ValueError(f"unknown mesh_type: {mesh_type}")
    if mesh_type == "tritter" and n_modes < 3:
        raise ValueError("tritter mesh requires n_modes >= 3")

    counter = itertools.count()

    def mesh() -> pcvl.Circuit:
        if mesh_type == "mzi":
            return _mzi_mesh(n_modes, counter)
        return _tritter_mesh(n_modes, n_modes, counter)

    c = pcvl.Circuit(n_modes)
    c.add(0, mesh(), merge=True)
    for i in range(n_modes):
        c.add(i, pcvl.PS(pcvl.P(f"x{i}")))
    c.add(0, mesh(), merge=True)
    return c
