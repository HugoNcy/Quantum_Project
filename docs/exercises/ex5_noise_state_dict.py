"""Exercise 5 -- Noise with losses and the state_dict trap.

Goal: see trap 3 fail for real, then fix it. A NoiseModel with
transmittance < 1 adds internal loss modes to the layer, so a full
load_state_dict from a lossless model fails. The fix: copy only the
trainable circuit phases, matched by parameter name.
"""

import torch
import perceval as pcvl
import merlin

FOCK = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)

def make_layer(noise=None):
    builder = merlin.CircuitBuilder(n_modes=4)
    builder.add_angle_encoding(name="x")
    builder.add_entangling_layer(model="mzi", trainable=True)
    return merlin.QuantumLayer(
        input_size=4,
        builder=builder,
        n_photons=2,
        measurement_strategy=FOCK,
        noise=noise,
    )

clean = make_layer()
lossy = make_layer(noise=pcvl.NoiseModel(transmittance=0.9))

print("clean state_dict keys:", sorted(clean.state_dict().keys()))
print("lossy state_dict keys:", sorted(lossy.state_dict().keys()))

# --- The trap: full state_dict transfer fails across noise profiles ---
try:
    lossy.load_state_dict(clean.state_dict())
    print("load_state_dict: unexpectedly succeeded")
except RuntimeError as e:
    print("load_state_dict FAILED as expected:\n", str(e)[:300])

# --- The fix: copy only trainable circuit phases, by name ---
def copy_circuit_params(src_layer, dst_layer):
    """Copy trainable circuit phases between layers with different noise.

    Never copy the full state_dict across noise profiles: lossy layers
    contain extra internal tensors (loss modes) that clean layers lack.
    """
    src = dict(src_layer.named_parameters())
    with torch.no_grad():
        for name, p in dst_layer.named_parameters():
            if name in src and src[name].shape == p.shape:
                p.copy_(src[name])

copy_circuit_params(clean, lossy)

# Verify: phases are now identical.
src = dict(clean.named_parameters())
ok = all(torch.equal(p, src[n]) for n, p in lossy.named_parameters() if n in src)
print("phases identical after copy_circuit_params:", ok)
