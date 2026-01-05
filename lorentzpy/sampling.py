"""Monte Carlo sampling on the Lorentz (hyperboloid) manifold.

This module provides functions for sampling points on the hyperbolic space
using user-defined distance distributions centered at weighted embedding points.
"""

import torch
import math
from typing import Optional, Callable, Tuple
import numpy as np

from . import minkowski
from . import hyperboloid


def project_to_tangent_space(
    mu: torch.Tensor, v: torch.Tensor, k: float = 1.0
) -> torch.Tensor:
    """Project ambient vector v onto tangent space at mu on the hyperboloid.

    Note: This is different from logmap. logmap maps a point ON the hyperboloid
    to the tangent space, while this function projects an arbitrary ambient vector
    (not necessarily on the hyperboloid) onto the tangent space.

    The tangent space T_mu H^d consists of vectors satisfying <v, mu>_L = 0.
    Formula: proj(v) = v + <v, mu>_L / k * mu

    Args:
        mu: Base point on hyperboloid, shape (..., d+1)
        v: Ambient vector to project, shape (..., d+1)
        k: Curvature parameter (k > 0)

    Returns:
        Projected tangent vector, shape (..., d+1)
    """
    inner = minkowski.bilinear_pairing(mu, v)
    return v + (inner / k).unsqueeze(-1) * mu


def sample_tangent_direction(
    mu: torch.Tensor, k: float = 1.0, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Sample uniformly distributed unit direction on tangent space at mu.

    Algorithm:
    1. Sample v ~ N(0, I) in ambient (d+1)-dimensional space
    2. Project v onto T_mu H^d (using project_to_tangent_space, not logmap)
    3. Normalize to unit Lorentzian norm (<u, u>_L = 1)

    Args:
        mu: Base points on hyperboloid, shape (batch, d+1)
        k: Curvature parameter
        generator: PyTorch random generator

    Returns:
        Unit tangent vectors, shape (batch, d+1)
    """
    # Sample in ambient space
    v = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)

    # Project onto tangent space at each mu
    # Note: We use project_to_tangent_space (not logmap) because v is not on the hyperboloid
    v_tangent = project_to_tangent_space(mu, v, k)

    # Normalize to unit Lorentzian norm
    # For tangent vectors, <v, v>_L > 0 (spacelike)
    norm_sq = minkowski.bilinear_pairing(v_tangent, v_tangent)
    norm = torch.sqrt(torch.clamp(norm_sq, min=1e-15))
    u = v_tangent / norm.unsqueeze(-1)

    return u


# =============================================================================
# Built-in distance distribution samplers
# =============================================================================


def half_cauchy_sampler(
    gamma: float,
) -> Callable[
    [int, torch.device, torch.dtype, Optional[torch.Generator]], torch.Tensor
]:
    """Create a half-Cauchy distance sampler.

    Density: p(d) ∝ 1/(d² + γ²) for d >= 0

    Args:
        gamma: Scale parameter

    Returns:
        Sampler function that takes (n_samples, device, dtype, generator) and returns distances
    """

    def sampler(
        n_samples: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        u = torch.rand(n_samples, device=device, dtype=dtype, generator=generator)
        return gamma * torch.tan(u * math.pi / 2)

    return sampler


def exponential_sampler(
    rate: float,
) -> Callable[
    [int, torch.device, torch.dtype, Optional[torch.Generator]], torch.Tensor
]:
    """Create an exponential distance sampler.

    Density: p(d) = rate * exp(-rate * d) for d >= 0

    Args:
        rate: Rate parameter (1/mean)

    Returns:
        Sampler function that takes (n_samples, device, dtype, generator) and returns distances
    """

    def sampler(
        n_samples: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        u = torch.rand(n_samples, device=device, dtype=dtype, generator=generator)
        return -torch.log(1 - u) / rate

    return sampler


def uniform_sampler(
    max_distance: float,
) -> Callable[
    [int, torch.device, torch.dtype, Optional[torch.Generator]], torch.Tensor
]:
    """Create a uniform distance sampler.

    Density: p(d) = 1/max_distance for 0 <= d <= max_distance

    Args:
        max_distance: Maximum distance

    Returns:
        Sampler function that takes (n_samples, device, dtype, generator) and returns distances
    """

    def sampler(
        n_samples: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return (
            torch.rand(n_samples, device=device, dtype=dtype, generator=generator)
            * max_distance
        )

    return sampler


def gaussian_sampler(
    sigma: float,
) -> Callable[
    [int, torch.device, torch.dtype, Optional[torch.Generator]], torch.Tensor
]:
    """Create a half-Gaussian (folded normal) distance sampler.

    Samples |X| where X ~ N(0, sigma^2)

    Args:
        sigma: Standard deviation

    Returns:
        Sampler function that takes (n_samples, device, dtype, generator) and returns distances
    """

    def sampler(
        n_samples: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return torch.abs(
            torch.randn(n_samples, device=device, dtype=dtype, generator=generator)
            * sigma
        )

    return sampler


# =============================================================================
# Main sampling function
# =============================================================================


def monte_carlo_sample(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    distance_sampler: Callable[
        [int, torch.device, torch.dtype, Optional[torch.Generator]], torch.Tensor
    ],
    k: float = 1.0,
    max_distance: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate Monte Carlo samples for density estimation on hyperboloid.

    Algorithm:
    1. Sample center mu with probability proportional to weights
    2. Sample uniform direction u on tangent space at mu
    3. Sample distance d from the provided distance_sampler
    4. Compute point via exp_mu(d * u) using hyperboloid.exp_unit_tangents

    Note: For distributions with heavy tails (e.g., Cauchy), use max_distance
    to prevent numerical overflow. For float32, max_distance=10.0 is safe.
    Without truncation, samples at very large distances may have numerical
    errors and will be filtered out.

    Args:
        embeddings: Points on hyperboloid, shape (N, d+1)
        weights: Sampling weights for each embedding, shape (N,)
        n_samples: Total number of samples to generate
        distance_sampler: Callable that takes (n_samples, device, dtype, generator)
            and returns distances as torch.Tensor of shape (n_samples,)
        k: Curvature parameter
        max_distance: Optional maximum distance to truncate samples.
            Recommended for heavy-tailed distributions to prevent overflow.
        generator: PyTorch random generator

    Returns:
        Sampled points on hyperboloid, shape (n_valid_samples, d+1)
        Note: May return fewer than n_samples if some samples have numerical errors.

    Example:
        >>> import lorentzpy
        >>> from lorentzpy.sampling import monte_carlo_sample, half_cauchy_sampler
        >>>
        >>> # Create embeddings
        >>> embeddings = lorentzpy.from_poincare(torch.randn(100, 2) * 0.3)
        >>> weights = torch.ones(100)
        >>>
        >>> # Sample with half-Cauchy distribution (use max_distance for stability)
        >>> sampler = half_cauchy_sampler(gamma=1.0)
        >>> samples = monte_carlo_sample(embeddings, weights, 1000, sampler, max_distance=10.0)
    """
    # Ensure inputs are tensors
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)

    device = embeddings.device
    dtype = embeddings.dtype

    if not torch.is_tensor(weights):
        weights = torch.as_tensor(weights, device=device, dtype=dtype)

    N = len(embeddings)

    # Normalize weights to probabilities
    probs = torch.clamp(weights, min=0)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = torch.ones(N, device=device, dtype=dtype) / N

    # Step 1: Sample centers with probability proportional to weights
    center_indices = torch.multinomial(
        probs, n_samples, replacement=True, generator=generator
    )
    centers = embeddings[center_indices]  # shape (n_samples, d+1)

    # Step 2: Sample uniform directions on tangent space
    unit_tangents = sample_tangent_direction(
        centers, k, generator
    )  # shape (n_samples, d+1)

    # Step 3: Sample distances from user-provided distribution
    distances = distance_sampler(n_samples, device, dtype, generator)

    # Optional: truncate distances
    if max_distance is not None:
        distances = torch.clamp(distances, max=max_distance)

    # Step 4: Apply exponential map using lorentzpy's exp_unit_tangents
    samples = hyperboloid.exp_unit_tangents(centers, unit_tangents, distances, k)

    # Project back onto hyperboloid for numerical stability
    # Large distances can cause floating point errors; this ensures samples
    # satisfy the hyperboloid constraint: -x_0^2 + ||x_spatial||^2 = -k
    samples = hyperboloid.update_time_coord(samples, k=k)

    # Filter out numerically invalid samples (NaN or Inf)
    valid_mask = torch.isfinite(samples).all(dim=-1)
    n_invalid = (~valid_mask).sum().item()
    if n_invalid > 0:
        import warnings

        warnings.warn(
            f"{n_invalid} samples had numerical errors and were filtered out. "
            f"Consider using max_distance parameter (e.g., max_distance=10.0) "
            f"to prevent overflow with heavy-tailed distributions.",
            RuntimeWarning,
        )
    samples = samples[valid_mask]

    return samples


def compute_density_monte_carlo(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    projector: Callable[[torch.Tensor], torch.Tensor],
    distance_sampler: Callable[
        [int, torch.device, torch.dtype, Optional[torch.Generator]], torch.Tensor
    ],
    k: float = 1.0,
    grid_size: int = 100,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    max_distance: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute density on a 2D grid using Monte Carlo sampling on hyperboloid.

    This function generates Monte Carlo samples on the hyperboloid manifold,
    projects them to a lower-dimensional space using the provided projector,
    converts to Poincare disk coordinates, and computes a 2D histogram density.

    Args:
        embeddings: Points on hyperboloid, shape (N, d+1)
        weights: Sampling weights for each embedding, shape (N,). Points are
            sampled with probability proportional to these weights.
        n_samples: Number of Monte Carlo samples to generate
        projector: Callable that projects hyperboloid samples to lower dimensions.
            Takes tensor of shape (n_samples, d+1) and returns tensor of shape
            (n_samples, 3) for 2D visualization. Use identity function if
            embeddings are already 3D.
        distance_sampler: Callable for sampling distances from center points.
            Takes (n_samples, device, dtype, generator) and returns distances.
            See half_cauchy_sampler, exponential_sampler, etc.
        k: Curvature parameter of the hyperboloid (default: 1.0)
        grid_size: Resolution of the output density grid (default: 100)
        xlim: Tuple of (min, max) for x-axis in Poincare coordinates.
            Defaults to (-0.99, 0.99).
        ylim: Tuple of (min, max) for y-axis in Poincare coordinates.
            Defaults to (-0.99, 0.99).
        max_distance: Maximum distance to truncate samples. Recommended for
            heavy-tailed distributions to prevent numerical overflow.
        generator: PyTorch random generator for reproducibility

    Returns:
        Tuple of (X_grid, Y_grid, density):
            - X_grid: x-coordinates meshgrid, shape (grid_size, grid_size)
            - Y_grid: y-coordinates meshgrid, shape (grid_size, grid_size)
            - density: Density values on grid, shape (grid_size, grid_size)

    Example:
        >>> import torch
        >>> import lorentzpy
        >>> from lorentzpy.sampling import (
        ...     compute_density_monte_carlo,
        ...     half_cauchy_sampler
        ... )
        >>>
        >>> # Create 3D hyperboloid embeddings (for 2D Poincare disk)
        >>> embeddings = lorentzpy.from_poincare(torch.randn(100, 2) * 0.3)
        >>> weights = torch.ones(100)
        >>>
        >>> # Identity projector for 3D embeddings
        >>> projector = lambda x: x
        >>>
        >>> # Compute density
        >>> X, Y, density = compute_density_monte_carlo(
        ...     embeddings, weights, n_samples=10000,
        ...     projector=projector,
        ...     distance_sampler=half_cauchy_sampler(gamma=1.0),
        ...     max_distance=10.0
        ... )
    """
    # Generate Monte Carlo samples
    samples = monte_carlo_sample(
        embeddings, weights, n_samples, distance_sampler, k, max_distance, generator
    )
    samples = projector(samples)

    # Convert samples to Poincare disk using lorentzpy's to_poincare
    if torch.is_tensor(samples):
        samples_poincare = hyperboloid.to_poincare(samples, k=k).cpu().numpy()
    else:
        samples_poincare = samples

    # Filter samples inside unit disk
    norms_sq = np.sum(samples_poincare**2, axis=1)
    mask = norms_sq < 1.0
    samples_poincare = samples_poincare[mask]

    # Set default limits
    if xlim is None:
        xlim = (-0.99, 0.99)
    if ylim is None:
        ylim = (-0.99, 0.99)

    # Create grid
    x_edges = np.linspace(xlim[0], xlim[1], grid_size + 1)
    y_edges = np.linspace(ylim[0], ylim[1], grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Use histogram
    density, _, _ = np.histogram2d(
        samples_poincare[:, 0],
        samples_poincare[:, 1],
        bins=[x_edges, y_edges],
        density=True,
    )

    X_grid, Y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")

    # Transpose for matplotlib compatibility
    return X_grid.T, Y_grid.T, density.T
