"""Internal utilities for lorentzpy."""

import torch


def orthonormal(Q):
    """Return orthonormal basis via SVD.

    Args:
        Q: torch.tensor of shape (k, d) - k basis vectors in d dimensions

    Returns:
        torch.tensor of shape (k, d) - orthonormalized basis vectors
    """
    u, s, v = torch.linalg.svd(Q, full_matrices=False)
    return v[:Q.shape[0], :]


def reflect(x, Q):
    """Reflect points (Euclidean) through space spanned by rows of Q.

    Args:
        x: torch.tensor of shape (n, d) - points to reflect
        Q: torch.tensor of shape (k, d) - basis spanning the reflection plane

    Returns:
        torch.tensor of shape (n, d) - reflected points
    """
    # Project x onto Q and reflect
    Q_ortho = orthonormal(Q)
    proj = x @ Q_ortho.T @ Q_ortho
    return 2 * proj - x
