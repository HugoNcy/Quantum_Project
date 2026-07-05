"""MMD loss with a mixture of gaussian kernels."""

import torch

DEFAULT_BANDWIDTHS = (0.1, 0.5, 1.0, 2.0, 5.0)


def mmd_loss(x: torch.Tensor, y: torch.Tensor, bandwidths=DEFAULT_BANDWIDTHS) -> torch.Tensor:
    """Squared MMD between samples x (generated) and y (target), shapes (B, D)."""

    def kernel(a, b):
        d2 = torch.cdist(a, b).pow(2)
        return sum(torch.exp(-d2 / (2 * s**2)) for s in bandwidths) / len(bandwidths)

    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()
