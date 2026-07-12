"""Synthetic 2D target distributions for the generative model."""

import torch


def gaussian(n: int, seed: int = 0) -> torch.Tensor:
    """Single 2D gaussian centered at (1, -1) with std 0.5."""
    # Dedicated generator so the dataset is reproducible independently of
    # the global torch RNG state.
    g = torch.Generator().manual_seed(seed)
    # Standard normal samples, scaled to std 0.5 and shifted to (1, -1).
    return torch.randn(n, 2, generator=g) * 0.5 + torch.tensor([1.0, -1.0])


def two_gaussians(n: int, seed: int = 0) -> torch.Tensor:
    """Balanced mixture of two 2D gaussians at (-1, 0) and (1, 0), std 0.3."""
    g = torch.Generator().manual_seed(seed)
    # The two mixture component centers.
    centers = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    # For each sample, pick component 0 or 1 with probability 1/2.
    idx = torch.randint(0, 2, (n,), generator=g)
    # Gaussian noise of std 0.3 around the selected center.
    return torch.randn(n, 2, generator=g) * 0.3 + centers[idx]


def ring(n: int, seed: int = 0) -> torch.Tensor:
    """Points on a unit circle with radial noise, std 0.1."""
    g = torch.Generator().manual_seed(seed)
    # Uniform angle on [0, 2pi).
    angles = torch.rand(n, generator=g) * 2 * torch.pi
    # Radius fluctuates around 1 with gaussian noise of std 0.1.
    radii = 1.0 + torch.randn(n, generator=g) * 0.1
    # Polar to cartesian, stacked into an (n, 2) tensor.
    return torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)


# Registry used by the training scripts to select a target by name.
DATASETS = {"gaussian": gaussian, "two_gaussians": two_gaussians, "ring": ring}


def log_returns(n: int, seed: int = 0) -> torch.Tensor:
    """Synthetic daily log-returns: standardized Student-t, df = 4 (phase 1bis).

    Stand-in for an empirical return series (no market data offline).
    A Student-t with 4 degrees of freedom has the leptokurtic shape of real
    daily returns. Standardizing to unit variance keeps DEFAULT_BANDWIDTHS
    valid (review F5) without touching the loss; standardization is affine,
    so the tail shape the phase is about is unchanged. Returns (n, 1).
    """
    g = torch.Generator().manual_seed(seed)
    df = 4
    # Student-t via its definition t = z / sqrt(chi2 / df); for integer df
    # the chi2 is a sum of df squared standard normals, all seeded by g
    # (torch.distributions samplers do not accept a generator).
    z = torch.randn(n, 1, generator=g)
    chi2 = torch.randn(n, df, generator=g).pow(2).sum(dim=1, keepdim=True)
    t = z / torch.sqrt(chi2 / df)
    # Exact variance of a t distribution is df / (df - 2); rescale to 1.
    return t / (df / (df - 2)) ** 0.5


# Registered after the 2D targets: this one is 1D, shape (n, 1).
DATASETS["log_returns"] = log_returns
