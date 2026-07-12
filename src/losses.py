"""MMD loss with a mixture of gaussian kernels."""

import torch

# Kernel scales chosen to bracket the typical inter-point distances of the
# 2D synthetic targets (order 1). Small scales see fine structure, large
# scales see the global shape.
DEFAULT_BANDWIDTHS = (0.1, 0.5, 1.0, 2.0, 5.0)


def mmd_loss(x: torch.Tensor, y: torch.Tensor, bandwidths=DEFAULT_BANDWIDTHS) -> torch.Tensor:
    """Squared MMD between samples x (generated) and y (target), shapes (B, D).

    Biased V-statistic estimator: the diagonal terms k(a, a) = 1 are kept in
    the means, so the value at equal distributions is positive and depends on
    the batch size. Comparisons are only valid at a fixed batch size.
    """

    def kernel(a, b):
        # Squared euclidean distances between all pairs of rows.
        d2 = torch.cdist(a, b).pow(2)
        # Average of gaussian kernels over the bandwidth mixture.
        return sum(torch.exp(-d2 / (2 * s**2)) for s in bandwidths) / len(bandwidths)

    # Standard expansion of MMD^2: E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)].
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()
