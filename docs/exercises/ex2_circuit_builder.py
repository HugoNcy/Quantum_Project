"""Exercise 2 -- MerLin CircuitBuilder: fluent circuit construction.

Goal: build the standard project circuit (angle encoding + trainable MZI mesh)
and inspect what the builder declares as input / trainable parameters.
"""

import merlin

builder = merlin.CircuitBuilder(n_modes=4)

# Input encoding: classical features become phase-shifter angles.
# The name defines the parameter prefix ("x0", "x1", ...).
builder.add_angle_encoding(name="x")

# Trainable universal mesh: Clements grid of MZIs.
builder.add_entangling_layer(model="mzi", trainable=True)

# Inspect what the builder produced (this is what QuantumLayer will infer).
print("builder:", builder)
print("input parameter prefixes:", builder.input_parameter_prefixes)
