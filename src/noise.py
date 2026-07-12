"""Noise profile grid and cross-profile weight transfer (plan, phase 2)."""

import math

import perceval as pcvl
import torch

# Named noise profiles of the project grid. Kept as kwargs dicts so they
# serialize directly into run logs; the NoiseModel is built on demand.
PROFILE_PARAMS = {
    "P0": {},
    "P1": {"indistinguishability": 0.95},
    "P2": {"indistinguishability": 0.90},
    "P3": {"indistinguishability": 0.85},
    "P4": {"indistinguishability": 0.95, "transmittance": 0.9},
}
PROFILE_ORDER = list(PROFILE_PARAMS)


def make_noise(name: str) -> pcvl.NoiseModel | None:
    """NoiseModel for a named profile; None for the clean reference P0."""
    params = PROFILE_PARAMS[name]
    return pcvl.NoiseModel(**params) if params else None


def is_lossy(name: str) -> bool:
    """True when the profile removes photons (extends the output space)."""
    params = PROFILE_PARAMS[name]
    return params.get("transmittance", 1.0) < 1.0 or params.get("brightness", 1.0) < 1.0


def expected_dim(n_modes: int, n_photons: int, lossy: bool) -> int:
    """Output dimension of a FOCK-forced layer, per docs/OFFLINE_KIT.md.

    Without losses: C(n+m-1, n) states with exactly n photons. Losses add
    every photon count k < n, so the dimension becomes the sum over k <= n.
    """
    if not lossy:
        return math.comb(n_photons + n_modes - 1, n_photons)
    return sum(math.comb(k + n_modes - 1, k) for k in range(n_photons + 1))


def _occupation(key) -> tuple:
    """Normalize an output_keys entry (tuple or BasicState) to a plain tuple."""
    return tuple(int(v) for v in key)


def transfer_generator(src_gen: torch.nn.Module, dst_gen: torch.nn.Module) -> None:
    """Move trained weights between generators with different noise profiles.

    Works on any generator exposing .quantum (a QuantumLayer) and .adapter
    (Sequential starting with a Linear over the Fock distribution), such as
    PhotonicGenerator and MeshGenerator from model.py.

    - Parameters with matching name and shape (encoder, circuit phases,
      adapter tail) are copied directly. Never a full state_dict load:
      lossy layers carry extra internal keys (plan, trap 3).
    - The first adapter Linear has one input column per Fock state, and a
      lossy profile has more states (photon counts k < n). Columns are
      aligned by occupation tuple through layer.output_keys; states the
      source never saw get zero columns (the transferred model is blind to
      loss events, which is the honest deployment semantics), and source
      states absent at destination are dropped.
    """
    src_params = dict(src_gen.named_parameters())
    with torch.no_grad():
        for name, p in dst_gen.named_parameters():
            # Copy every same-named parameter whose shape matches.
            if name in src_params and src_params[name].shape == p.shape:
                p.copy_(src_params[name])
        src_w = src_gen.adapter[0].weight
        dst_w = dst_gen.adapter[0].weight
        if src_w.shape != dst_w.shape:
            # Column j of the adapter weight multiplies the probability of
            # Fock state output_keys[j]; match columns by occupation tuple.
            src_index = {
                _occupation(k): i for i, k in enumerate(src_gen.quantum.output_keys)
            }
            dst_keys = [_occupation(k) for k in dst_gen.quantum.output_keys]
            matched = sum(k in src_index for k in dst_keys)
            if matched == 0:
                raise ValueError("no common Fock states between the two layers")
            dst_w.zero_()
            for j, key in enumerate(dst_keys):
                i = src_index.get(key)
                if i is not None:
                    dst_w[:, j] = src_w[:, i]
