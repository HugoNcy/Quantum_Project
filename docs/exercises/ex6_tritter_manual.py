"""Exercise 6 (bonus) -- Hand-built tritter passed to QuantumLayer.

Goal: rehearse the project's main API friction point (Phase 3). Build a
3-mode circuit in pure Perceval containing:
  - input encoding phase shifters named "x0".."x2",
  - a fixed tritter (3x3 DFT unitary, decomposed into BS+PS),
  - trainable phase shifters named "theta0".."theta2",
  - a second tritter.
Then wrap it in a QuantumLayer with MANUAL parameter declaration
(input_parameters / trainable_parameters by name prefix) -- the exception
to the "pass the builder" rule.

Validated API facts (perceval 1.2.3):
  - Circuit.decomposition needs a FULL MZI template (BS//PS//BS//PS);
    a bare BS template fails and the function returns None.
  - phase_shifter_fn receives the SOLVED PHASE VALUE (a float), not an
    index: `lambda phi: pcvl.PS(phi)` yields a fully numeric circuit.
  - pcvl.Unitary(matrix) is the no-decomposition shortcut, fine for
    simulation but hides the physical BS/PS layout.
"""

import numpy as np
import torch
import perceval as pcvl
import merlin

def dft3_unitary():
    """The ideal tritter: 3x3 DFT matrix, coefficients in cube roots of unity."""
    w = np.exp(2j * np.pi / 3)
    u = np.array([[1, 1, 1],
                  [1, w, w**2],
                  [1, w**2, w]]) / np.sqrt(3)
    return pcvl.MatrixN(u)

def tritter_circuit():
    """Fixed DFT3 decomposed into BS + PS (hardware-realistic layout)."""
    mzi = pcvl.BS() // pcvl.PS(pcvl.P("t")) // pcvl.BS() // pcvl.PS(pcvl.P("p"))
    decomposed = pcvl.Circuit.decomposition(
        dft3_unitary(),
        mzi,
        phase_shifter_fn=lambda phi: pcvl.PS(phi),  # phi is the solved value
        shape="triangle",
    )
    assert decomposed is not None, "decomposition failed"
    assert not decomposed.get_parameters(), "tritter must have no free parameters"
    return decomposed

# Sanity check: the decomposed circuit reproduces the DFT3 matrix.
assert np.allclose(np.asarray(tritter_circuit().compute_unitary()),
                   np.asarray(dft3_unitary()), atol=1e-5)

# Assemble: encoding PS -> tritter -> trainable PS -> tritter
circuit = pcvl.Circuit(3, name="tritter_block")
for i in range(3):
    circuit.add(i, pcvl.PS(phi=pcvl.P(f"x{i}")))
circuit.add(0, tritter_circuit(), merge=True)
for i in range(3):
    circuit.add(i, pcvl.PS(phi=pcvl.P(f"theta{i}")))
circuit.add(0, tritter_circuit(), merge=True)

print("free parameters:", [p.name for p in circuit.get_parameters()])

# Manual declaration: this is the tritter exception to trap 1.
layer = merlin.QuantumLayer(
    circuit=circuit,                    # raw pcvl.Circuit
    input_size=3,
    input_parameters=["x"],             # prefix -> x0, x1, x2
    trainable_parameters=["theta"],     # prefix -> theta0, theta1, theta2
    n_photons=2,
    measurement_strategy=merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK),
)

x = torch.rand(8, 3) * torch.pi
out = layer(x)
print("output shape:", tuple(out.shape), "-> C(4,2) = 6 states (3 modes, 2 photons, FOCK)")
out[:, 0].mean().backward()   # non-degenerate dummy loss (see exercise 4)
print("params with grads:",
      [(n, round(p.grad.norm().item(), 6)) for n, p in layer.named_parameters()
       if p.grad is not None])
