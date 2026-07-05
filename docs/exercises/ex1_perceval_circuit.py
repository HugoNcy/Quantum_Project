"""Exercise 1 -- Pure Perceval: build a small circuit from BS and PS.

Goal: get comfortable with pcvl.Circuit, components and named parameters.
This is the exact skill needed later to hand-build the tritter mesh.
"""

import numpy as np
import perceval as pcvl

# A 3-mode circuit: BS on modes (0,1), a parametrized PS on mode 1,
# then BS on modes (1,2).
circuit = pcvl.Circuit(3, name="warmup")
circuit.add((0, 1), pcvl.BS())
circuit.add(1, pcvl.PS(phi=pcvl.P("phi0")))
circuit.add((1, 2), pcvl.BS())

# Named parameters are the interface used later by MerLin (prefix matching).
params = circuit.get_parameters()
print("parameter names:", [p.name for p in params])

# Assign a value to the symbolic parameter, then compute the 3x3 unitary.
params[0].set_value(np.pi / 4)
u = circuit.compute_unitary()
print("unitary:\n", np.round(u, 3))
print("is unitary:", np.allclose(u @ np.conjugate(u.T), np.eye(3)))
