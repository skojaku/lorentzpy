"""Centering operations for hyperbolic spaces with curvature support.

Functions for centering data on the hyperboloid by moving the Frechet mean to the origin.
"""

import torch

from . import hyperboloid
from .frechet import frechet_mean


def _reflection_center(mu):
    """Compute center of inversion circle.

    Args:
        mu: torch.tensor of shape (..., dim) - point in Poincare ball

    Returns:
        torch.tensor of shape (..., dim) - center of inversion circle
    """
    return mu / torch.sum(mu**2, dim=-1, keepdim=True).clamp_min(1e-15)


def _isometric_transform(x, a):
    """Circle inversion of x through orthogonal circle centered at a.

    This is an isometry of the Poincare ball that maps a to the origin.

    Args:
        x: torch.tensor of shape (..., dim) - points in Poincare ball
        a: torch.tensor of shape (..., dim) - center of inversion circle

    Returns:
        torch.tensor of shape (..., dim) - transformed points
    """
    r2 = torch.sum(a**2, dim=-1, keepdim=True) - 1.0
    u = x - a
    return r2 / torch.sum(u**2, dim=-1, keepdim=True).clamp_min(1e-15) * u + a


def _reflect_at_zero(x, mu):
    """Map x under the isometry (inversion) taking mu to origin.

    Args:
        x: torch.tensor of shape (n, dim) - points in Poincare ball
        mu: torch.tensor of shape (dim,) - point to map to origin

    Returns:
        torch.tensor of shape (n, dim) - transformed points
    """
    mu_norm_sq = torch.sum(mu**2, dim=-1)

    # If mean is already at origin, no transformation needed
    if mu_norm_sq.max() < 1e-10:
        return x

    a = _reflection_center(mu)
    return _isometric_transform(x, a)


def center(x, k=1.0, mu=None, lr=0.1, eps=1e-5, max_steps=5000, return_mean=False, return_converged=False):
    """Center data to have Frechet mean at the origin.

    Applies an isometry that moves the Frechet mean to the origin of the hyperboloid.

    Args:
        x: torch.tensor of shape (n, dim+1) - hyperboloid coordinates
        k: float, curvature parameter (default: 1.0)
        mu: torch.tensor of shape (1, dim+1), pre-computed Frechet mean (optional)
        lr: float, learning rate for mean computation if mu not provided
        eps: float, convergence threshold for mean computation
        max_steps: int, maximum steps for mean computation
        return_mean: bool, if True, also return the computed Frechet mean
        return_converged: bool, if True, also return convergence status

    Returns:
        x_centered: torch.tensor of shape (n, dim+1) - centered hyperboloid coordinates
        mu: torch.tensor (only if return_mean=True) - Frechet mean before centering
        has_converged: bool (only if return_converged=True)
    """
    # Compute Frechet mean if not provided
    if mu is None:
        mu_hyperboloid, has_converged = frechet_mean(
            x, k=k, lr=lr, eps=eps, max_steps=max_steps, return_converged=True
        )
    else:
        mu_hyperboloid = mu
        has_converged = True

    # Convert to Poincare ball
    x_poincare = hyperboloid.to_poincare(x, k)
    mu_poincare = hyperboloid.to_poincare(mu_hyperboloid, k).squeeze(0)

    # Apply isometric transformation to move mean to origin
    x_centered_poincare = _reflect_at_zero(x_poincare, mu_poincare)

    # Convert back to hyperboloid
    x_centered = hyperboloid.from_poincare(x_centered_poincare, k)

    # Build return tuple based on options
    result = [x_centered]
    if return_mean:
        result.append(mu_hyperboloid)
    if return_converged:
        result.append(has_converged)

    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)


def center_with_mean(x, mu, k=1.0):
    """Center data given a pre-computed Frechet mean.

    Convenience function that applies the centering isometry directly.

    Args:
        x: torch.tensor of shape (n, dim+1) - hyperboloid coordinates
        mu: torch.tensor of shape (1, dim+1) - pre-computed Frechet mean
        k: float, curvature parameter (default: 1.0)

    Returns:
        x_centered: torch.tensor of shape (n, dim+1) - centered hyperboloid coordinates
    """
    return center(x, k=k, mu=mu, return_mean=False, return_converged=False)
