"""Exercise 4 -- PyTorch autograd through the quantum layer.

Goal: verify that gradients flow through the photonic simulation exactly like
through any nn.Module: forward on a batch, dummy loss, backward, inspect the
gradients of the quantum (phase) parameters.

Bonus pitfall discovered while validating this exercise: out.mean() is a
DEGENERATE dummy loss here. The layer outputs probability rows summing to 1,
so the mean over all entries is the constant 1/D and its gradient is exactly
zero. Use any loss that actually depends on the distribution shape.
"""

import torch
import merlin

builder = merlin.CircuitBuilder(n_modes=4)
builder.add_angle_encoding(name="x")
builder.add_entangling_layer(model="mzi", trainable=True)

layer = merlin.QuantumLayer(
    input_size=4,
    builder=builder,
    n_photons=2,
    measurement_strategy=merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK),
)

x = torch.rand(8, 4) * torch.pi   # batch of 8 samples, 4 features
out = layer(x)                    # (8, 10) probability rows

# --- Degenerate loss: rows sum to 1, so out.mean() == 1/10 is constant ---
loss_bad = out.mean()
loss_bad.backward(retain_graph=True)
grads_bad = {n: p.grad.norm().item() for n, p in layer.named_parameters()}
print("grad norms with out.mean() (constant loss):", grads_bad)  # ~0.0

# --- Proper dummy loss: probability of the first Fock state ---
layer.zero_grad()
loss = out[:, 0].mean()
loss.backward()
for name, p in layer.named_parameters():
    print(f"{name}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}, "
          f"grad_norm={round(p.grad.norm().item(), 6)}")
