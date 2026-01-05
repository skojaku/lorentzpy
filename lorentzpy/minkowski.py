"""Utility functions for the Minkowski metric with curvature support.

The Minkowski space has signature (-1, 1, 1, ...).
Points on the hyperboloid satisfy: -x_0^2 + x_1^2 + ... + x_d^2 = -k
where k > 0 is the curvature parameter (magnitude of negative curvature).
"""

import torch


MIN_NORM = 1e-15


def bilinear_pairing(x, y):
    """Compute the Minkowski bilinear pairing (dot product) of x and y.

    Uses signature (-1, 1, 1, ...):
        <x, y>_L = -x_0*y_0 + x_1*y_1 + x_2*y_2 + ...

    Args:
        x, y: torch.tensor of the same shape (..., dim), where dim >= 2

    Returns:
        torch.tensor of shape (...)
    """
    eucl_pairing = torch.sum(x * y, dim=-1, keepdim=False)
    return eucl_pairing - 2 * x[..., 0] * y[..., 0]


def squared_norm(x):
    """Compute the squared Minkowski norm of x.

    Args:
        x: torch.tensor of shape (..., dim)

    Returns:
        torch.tensor of shape (...)
    """
    return bilinear_pairing(x, x)


def pairwise_bilinear_pairing(x, y):
    """Compute the pairwise Minkowski bilinear pairings of two batches of vectors.

    Args:
        x: torch.tensor of shape (M, dim), where dim >= 2
        y: torch.tensor of shape (N, dim), where dim >= 2

    Returns:
        torch.tensor of shape (M, N)
    """
    return x @ y.T - 2 * torch.outer(x[:, 0], y[:, 0])


def orthogonal_projection(basis, x):
    """Compute the orthogonal projection of x onto the subspace spanned by basis.

    Uses Minkowski metric for orthogonality.

    Args:
        basis: torch.tensor of shape (num_basis, dim), where dim >= 2
        x: torch.tensor of shape (batch_size, dim), where dim >= 2

    Returns:
        torch.tensor of shape (batch_size, dim)

    Warning:
        Will not work if the subspace is tangent to the light cone.
    """
    A = pairwise_bilinear_pairing(basis, basis)
    B = pairwise_bilinear_pairing(basis, x)
    coefs = torch.linalg.solve(A, B)
    return coefs.T @ basis


def reflection(subspace, x, subspace_given_by_normal=True):
    """Compute the reflection of x through a linear subspace.

    The subspace has dimension 1 less than the ambient space.
    Uses Minkowski metric for orthogonality.

    Args:
        subspace: If subspace_given_by_normal:
                      torch.tensor of shape (dim,) - normal vector
                  Else:
                      torch.tensor of shape (dim-1, dim) - basis vectors
        x: torch.tensor of shape (batch_size, dim)
        subspace_given_by_normal: bool, how subspace is specified

    Returns:
        torch.tensor of shape (batch_size, dim)

    Warning:
        Will not work if the subspace is tangent to the light cone.
    """
    if subspace_given_by_normal:
        return x - 2 * orthogonal_projection(subspace.unsqueeze(0), x)
    else:
        return 2 * orthogonal_projection(subspace, x) - x
