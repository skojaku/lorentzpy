"""Frechet statistics for hyperbolic spaces with curvature support.

Functions for computing Frechet mean and variance on the hyperboloid.
"""

import torch

from . import hyperboloid


def frechet_mean(x, k=1.0, lr=0.1, eps=1e-5, max_steps=5000, return_converged=False):
    """Compute the Frechet mean of points on the hyperboloid.

    The Frechet mean minimizes the sum of squared hyperbolic distances.
    Computation is performed in the Poincare ball model for numerical stability,
    then converted back to hyperboloid coordinates.

    Args:
        x: torch.tensor of shape (n, dim+1) - hyperboloid coordinates
        k: float, curvature parameter (default: 1.0)
        lr: float, initial learning rate for gradient descent (default: 0.1)
        eps: float, convergence threshold for gradient norm (default: 1e-5)
        max_steps: int, maximum optimization steps (default: 5000)
        return_converged: bool, if True, also return convergence status

    Returns:
        mu: torch.tensor of shape (1, dim+1) - Frechet mean in hyperboloid coords
        has_converged: bool (only if return_converged=True)
    """
    # Convert to Poincare ball for computation
    x_poincare = hyperboloid.to_poincare(x, k)

    # Initialize with Euclidean mean projected to ball
    mu = torch.mean(x_poincare, dim=0, keepdim=True)
    mu = hyperboloid._poincare_project(mu, k)

    has_converged = False

    for step in range(max_steps):
        # Compute gradient: mean of log maps from mu to each point
        log_x = hyperboloid._poincare_logmap(mu, x_poincare, k)
        grad = torch.mean(log_x, dim=0, keepdim=True)

        # Adaptive learning rate based on gradient norm
        grad_norm = grad.norm(dim=-1, p=2).item()

        if grad_norm < eps:
            has_converged = True
            break

        # Use gradient norm to set step size (normalized gradient descent)
        current_lr = min(lr, 0.5 / (grad_norm + 1e-8))

        # Take step in direction of gradient
        delta_mu = current_lr * grad
        mu_new = hyperboloid._poincare_expmap(mu, delta_mu, k)
        mu_new = hyperboloid._poincare_project(mu_new, k)

        # Check for convergence based on movement
        movement = (mu_new - mu).norm().item()
        if movement < eps * 0.01:
            has_converged = True
            mu = mu_new
            break

        mu = mu_new

    # Convert back to hyperboloid
    mu_hyperboloid = hyperboloid.from_poincare(mu, k)

    if return_converged:
        return mu_hyperboloid, has_converged
    else:
        return mu_hyperboloid


def frechet_variance(x, k=1.0, mu=None, lr=0.1, eps=1e-5, max_steps=5000, return_converged=False):
    """Compute the Frechet variance of points on the hyperboloid.

    Frechet variance is the mean squared distance from the Frechet mean.

    Args:
        x: torch.tensor of shape (n, dim+1) - hyperboloid coordinates
        k: float, curvature parameter (default: 1.0)
        mu: torch.tensor of shape (1, dim+1), pre-computed Frechet mean (optional)
        lr: float, learning rate for mean computation if mu not provided
        eps: float, convergence threshold for mean computation
        max_steps: int, maximum steps for mean computation
        return_converged: bool, if True, also return convergence status

    Returns:
        var: torch.tensor scalar - Frechet variance
        has_converged: bool (only if return_converged=True)
    """
    if mu is None:
        mu_hyperboloid, has_converged = frechet_mean(
            x, k=k, lr=lr, eps=eps, max_steps=max_steps, return_converged=True
        )
    else:
        mu_hyperboloid = mu
        has_converged = True

    distances = hyperboloid.distance(x, mu_hyperboloid, k) ** 2
    var = torch.mean(distances)

    if return_converged:
        return var, has_converged
    else:
        return var
