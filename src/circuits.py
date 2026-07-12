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
    # Primitive cube root of unity, w = exp(2i pi / 3).
    w = np.exp(2j * np.pi / 3)
    # DFT matrix rows are powers of w; 1/sqrt(3) makes it unitary.
    u = np.array([[1, 1, 1], [1, w, w * w], [1, w * w, w]]) / np.sqrt(3)
    return pcvl.Matrix(u)


def tritter() -> pcvl.Circuit:
    """Fixed 3-mode circuit equal to the DFT 3x3 up to a global phase.

    Decomposed into beam splitters and phase shifters by Reck (triangle
    shape) on 3 modes. All phases are fixed numeric values, so none of
    them is picked up by the "x" or "theta" prefixes.
    """
    # Template cell used by the decomposition: BS, PS, BS, PS with two
    # free angles that the solver tunes for each cell of the triangle.
    mzi = (
        pcvl.BS()
        // (0, pcvl.PS(phi=pcvl.P("phi_a")))
        // pcvl.BS()
        // (0, pcvl.PS(phi=pcvl.P("phi_b")))
    )
    # Solve for the cell phases realizing the DFT unitary; the returned
    # circuit has numeric phases only (no free parameters left).
    return pcvl.Circuit.decomposition(
        dft3_matrix(),
        mzi,
        phase_shifter_fn=pcvl.PS,
        shape=pcvl.InterferometerShape.TRIANGLE,
    )


def _mzi_mesh(n_modes: int, counter) -> pcvl.Circuit:
    """Rectangular (Clements) mesh of MZIs with 2 trainable phases each."""

    def mzi(_: int) -> pcvl.Circuit:
        # One MZI cell: BS, trainable internal phase, BS, trainable
        # external phase. The counter keeps theta names globally unique.
        c = pcvl.Circuit(2)
        c.add(0, pcvl.BS())
        c.add(0, pcvl.PS(pcvl.P(f"theta{next(counter)}")))
        c.add(0, pcvl.BS())
        c.add(0, pcvl.PS(pcvl.P(f"theta{next(counter)}")))
        return c

    # Clements layout: n_modes*(n_modes-1)/2 cells on alternating mode
    # pairs, which is the universal mesh at optimal depth.
    return pcvl.GenericInterferometer(
        n_modes, mzi, shape=pcvl.InterferometerShape.RECTANGLE
    )


def _tritter_ps_modes(offset: int, n_modes: int) -> list[int]:
    """Modes receiving a trainable phase before a tritter layer at this offset.

    Tritters sit on consecutive triplets starting at the offset; a trailing
    group of fewer than 3 modes is dropped (no wraparound, which would not
    be planar on a chip). Within each triplet the first mode carries no
    phase shifter: a tritter is transparent to a common phase on its three
    modes, so one mode per triplet must serve as the phase reference,
    otherwise that common direction is a flat (gauge) parameter direction.
    """
    modes: list[int] = []
    for start in range(offset, n_modes - 2, 3):
        modes.extend((start + 1, start + 2))
    return modes


def _tritter_layers_for_budget(n_modes: int) -> int:
    """Number of tritter layers giving an effective budget close to the MZI.

    The comparison criterion is the measured Jacobian rank of the output
    probabilities with respect to the phases (review F1), not the raw
    parameter count. At the reference size n_modes=6, 12 layers give 56
    nominal phases and a measured rank of 49 to 50 depending on the random
    phase draw (one direction is numerically marginal), against 50 of 60
    for the MZI mesh; tests/test_review_findings.py enforces the parity. Other
    sizes fall back to matching the nominal MZI budget and need their rank
    re-measured before being used in a comparison.
    """
    if n_modes == 6:
        return 12
    target = n_modes * (n_modes - 1)
    offsets = [o for o in (0, 1, 2) if o + 3 <= n_modes]
    total = 0
    n_layers = 1
    # Phase shifters sit before layers 1 .. n_layers-1, so each extra
    # layer adds the phase count of its own placement pattern.
    while total < target:
        total += len(_tritter_ps_modes(offsets[n_layers % len(offsets)], n_modes))
        n_layers += 1
    return n_layers


def _tritter_mesh(n_modes: int, n_layers: int, counter) -> pcvl.Circuit:
    """Layers of tritters on shifted triplets with trainable phases between.

    Layer offsets cycle over the values that fit at least one tritter.
    At n_modes=6 the coverage is uneven by construction: offset 0 places
    tritters (0,1,2) and (3,4,5), but offsets 1 and 2 only fit one tritter,
    (1,2,3) and (2,3,4), so modes 0 and 5 are mixed in 1 layer out of 3
    while modes 2 and 3 are mixed in every layer. This center bias is a
    property of a planar tritter mesh without mode crossings, kept
    deliberately (review F2) and reported as such.

    Trainable phase shifters are placed before each layer, only on the
    modes that this layer mixes, minus one reference mode per triplet
    (review F1). Phases on unmixed modes would collapse pairwise or end as
    per-mode output phases invisible to photon counting, and the common
    phase of a triplet slides through its tritter unchanged; both cases
    are flat parameter directions that would inflate the nominal budget
    without adding expressivity.
    """
    offsets = [o for o in (0, 1, 2) if o + 3 <= n_modes]
    # One decomposed tritter block, copied at every position.
    base = tritter()
    c = pcvl.Circuit(n_modes)
    for layer in range(n_layers):
        offset = offsets[layer % len(offsets)]
        # No phases before layer 0: they would act on the fixed input
        # Fock state (or on the encoding phases) as pure gauge.
        if layer > 0:
            for i in _tritter_ps_modes(offset, n_modes):
                c.add(i, pcvl.PS(pcvl.P(f"theta{next(counter)}")))
        # Place the tritters of this layer on their triplets.
        for start in range(offset, n_modes - 2, 3):
            c.add(start, base.copy(), merge=True)
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

    # Single counter shared by both meshes so theta names never repeat.
    counter = itertools.count()

    def mesh() -> pcvl.Circuit:
        if mesh_type == "mzi":
            return _mzi_mesh(n_modes, counter)
        # Layer count tuned so the nominal budget matches the MZI mesh;
        # effective-rank parity is checked by tests/test_review_findings.py.
        return _tritter_mesh(n_modes, _tritter_layers_for_budget(n_modes), counter)

    c = pcvl.Circuit(n_modes)
    # First entangling mesh: mixes the input Fock state before encoding.
    c.add(0, mesh(), merge=True)
    # Angle encoding: one data phase per mode, names x0 .. x{n-1}, inserted
    # in mode order (the order, not the names, is what MerLin maps inputs by).
    for i in range(n_modes):
        c.add(i, pcvl.PS(pcvl.P(f"x{i}")))
    # Second entangling mesh: makes the encoding phases observable.
    c.add(0, mesh(), merge=True)
    return c
