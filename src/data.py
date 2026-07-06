"""Synthetic 2D target distributions for the generative model."""

import torch


def gaussian(n: int, seed: int = 0) -> torch.Tensor:
    """Single 2D gaussian centered at (1, -1) with std 0.5."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 2, generator=g) * 0.5 + torch.tensor([1.0, -1.0])


def two_gaussians(n: int, seed: int = 0) -> torch.Tensor:
    """Balanced mixture of two 2D gaussians at (-1, 0) and (1, 0), std 0.3."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    idx = torch.randint(0, 2, (n,), generator=g)
    return torch.randn(n, 2, generator=g) * 0.3 + centers[idx]


def ring(n: int, seed: int = 0) -> torch.Tensor:
    """Points on a unit circle with radial noise, std 0.1."""
    g = torch.Generator().manual_seed(seed)
    angles = torch.rand(n, generator=g) * 2 * torch.pi
    radii = 1.0 + torch.randn(n, generator=g) * 0.1
    return torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)


DATASETS = {"gaussian": gaussian, "two_gaussians": two_gaussians, "ring": ring}
