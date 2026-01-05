"""Lorentzian K-Nearest Neighbors using FAISS.

Efficient KNN search in hyperbolic space using the Lorentz model.
"""

import numpy as np
import faiss


class LorentzKNN:
    """K-Nearest Neighbors search using Lorentzian distance.

    Uses FAISS inner product index to compute Lorentzian distances efficiently.
    The Lorentzian inner product <x, y>_L = -x_0*y_0 + x_1*y_1 + ... is computed
    by negating the time coordinate before using FAISS's dot product.
    """

    def __init__(self, k=1.0):
        """Initialize LorentzKNN.

        Args:
            k: float, curvature parameter (default: 1.0)
        """
        self.k = k
        self.index = None

    def add(self, embeddings):
        """Build FAISS index from embeddings.

        Args:
            embeddings: np.ndarray of shape (n, dim+1) - Lorentz hyperboloid coordinates
                Can also be a torch.Tensor which will be converted to numpy.

        Returns:
            self for method chaining
        """
        # Convert to numpy if needed
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.detach().cpu().numpy()

        emb = embeddings.astype(np.float32).copy()

        # Negate time coordinate so that FAISS inner product gives -<x,y>_L
        # FAISS computes x @ y, but we want -x_0*y_0 + x_i*y_i
        # By negating x_0, we get: -x_0*y_0 + x_i*y_i = <x,y>_L (with negated x_0)
        emb[:, 0] *= -1

        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        return self

    def search(self, embeddings, n_neighbors):
        """Find k nearest neighbors.

        Args:
            embeddings: np.ndarray of shape (n, dim+1) - query points in Lorentz coords
                Can also be a torch.Tensor which will be converted to numpy.
            n_neighbors: int, number of neighbors to find

        Returns:
            distances: np.ndarray of shape (n, n_neighbors) - hyperbolic distances
            indices: np.ndarray of shape (n, n_neighbors) - neighbor indices
        """
        if self.index is None:
            raise ValueError("Must call add() before search()")

        # Convert to numpy if needed
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.detach().cpu().numpy()

        emb = embeddings.astype(np.float32).copy()

        # FAISS will compute query @ index (with index already having negated time)
        # This gives us the Lorentzian inner product <query, indexed>_L
        inner, indices = self.index.search(emb, n_neighbors)

        # Convert inner product to distance
        # d(x, y) = sqrt(k) * arcosh(-<x,y>_L / k)
        # Note: inner from FAISS is -<x,y>_L (because we negated time in index)
        inner_normalized = -inner.astype(np.float64) / self.k

        # Ensure argument to arcosh is >= 1 for numerical stability
        distances = np.sqrt(self.k) * np.arccosh(np.maximum(inner_normalized, 1.0))

        # Handle edge case where inner_normalized is very close to 1 (same point)
        mask = np.isclose(inner_normalized, 1.0)
        distances[mask] = 0.0

        return distances, indices

    def __repr__(self):
        n_points = self.index.ntotal if self.index is not None else 0
        return f"LorentzKNN(k={self.k}, n_points={n_points})"
