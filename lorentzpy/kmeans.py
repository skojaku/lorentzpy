"""Lorentzian K-Means clustering using Frechet mean.

K-means clustering on the Lorentz hyperboloid model using geodesic distances
and Frechet means as cluster centroids.
"""

import torch
import numpy as np
import math
from typing import Optional, Tuple, Union

from . import hyperboloid, minkowski
from .frechet import frechet_mean


class LorentzKMeans:
    """K-Means clustering on the Lorentz hyperboloid.

    Uses hyperbolic (geodesic) distances for cluster assignment and
    Frechet means for centroid computation. API follows scikit-learn conventions.

    Algorithm:
        1. Initialize centroids (k-means++ or random)
        2. Assign each point to nearest centroid using hyperbolic distance
        3. Update centroids as Frechet mean of assigned points
        4. Repeat until convergence

    Attributes (after fitting):
        cluster_centers_: torch.Tensor of shape (n_clusters, dim+1)
            Cluster centroids on the hyperboloid.
        labels_: torch.Tensor of shape (n_samples,)
            Labels of each point.
        inertia_: float
            Sum of squared distances to closest cluster center.
        n_iter_: int
            Number of iterations run.

    Example:
        >>> import torch
        >>> import lorentzpy
        >>>
        >>> # Create embeddings on hyperboloid
        >>> poincare_pts = torch.randn(1000, 8) * 0.3
        >>> poincare_pts = poincare_pts / (1 + poincare_pts.norm(dim=-1, keepdim=True))
        >>> embeddings = lorentzpy.from_poincare(poincare_pts)
        >>>
        >>> # Cluster
        >>> kmeans = lorentzpy.LorentzKMeans(n_clusters=10)
        >>> labels = kmeans.fit_predict(embeddings)
        >>> centroids = kmeans.cluster_centers_
    """

    def __init__(
        self,
        n_clusters: int,
        k: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 10,
        init: str = "k-means++",
        random_state: Optional[int] = None,
        frechet_lr: float = 0.1,
        frechet_max_steps: int = 100,
        verbose: bool = False,
    ):
        """Initialize LorentzKMeans.

        Args:
            n_clusters: int, number of clusters
            k: float, curvature parameter (default: 1.0)
            max_iter: int, maximum iterations per initialization (default: 100)
            tol: float, convergence tolerance for centroid movement (default: 1e-4)
            n_init: int, number of initializations to try (default: 10)
            init: str, initialization method - "k-means++" or "random" (default: "k-means++")
            random_state: Optional[int], random seed for reproducibility
            frechet_lr: float, learning rate for Frechet mean computation (default: 0.1)
            frechet_max_steps: int, max steps for Frechet mean (default: 100)
            verbose: bool, if True, print progress (default: False)
        """
        self.n_clusters = n_clusters
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state
        self.frechet_lr = frechet_lr
        self.frechet_max_steps = frechet_max_steps
        self.verbose = verbose

        # Fitted attributes (sklearn naming convention)
        self.cluster_centers_: Optional[torch.Tensor] = None
        self.labels_: Optional[torch.Tensor] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: int = 0

    def _compute_distances(
        self, x: torch.Tensor, centroids: torch.Tensor
    ) -> torch.Tensor:
        """Compute hyperbolic distances from points to centroids.

        Args:
            x: torch.Tensor of shape (n, dim+1) - points on hyperboloid
            centroids: torch.Tensor of shape (n_clusters, dim+1) - centroids

        Returns:
            torch.Tensor of shape (n, n_clusters) - distance matrix
        """
        # Compute pairwise distances efficiently
        # x: (n, dim+1), centroids: (n_clusters, dim+1)
        inner = minkowski.pairwise_bilinear_pairing(x, centroids)

        # d(x, y) = sqrt(k) * arcosh(-<x, y>_L / k)
        # Clamp input to acosh to be >= 1.0 for numerical stability
        return math.sqrt(self.k) * torch.acosh(
            torch.clamp(-inner / self.k, min=1.0 + 1e-15)
        )

    def _assign_clusters(
        self, x: torch.Tensor, centroids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign points to nearest centroids.

        Args:
            x: torch.Tensor of shape (n, dim+1) - points on hyperboloid
            centroids: torch.Tensor of shape (n_clusters, dim+1) - centroids

        Returns:
            labels: torch.Tensor of shape (n,) - cluster assignments
            distances: torch.Tensor of shape (n,) - distances to assigned centroids
        """
        dist_matrix = self._compute_distances(x, centroids)
        distances, labels = torch.min(dist_matrix, dim=1)
        return labels, distances

    def _update_centroids(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Update centroids as Frechet means of assigned points.

        Args:
            x: torch.Tensor of shape (n, dim+1) - points on hyperboloid
            labels: torch.Tensor of shape (n,) - cluster assignments

        Returns:
            torch.Tensor of shape (n_clusters, dim+1) - updated centroids
        """
        device = x.device
        dim = x.shape[1] - 1  # Poincare dimension

        # Convert to Poincare ball for computation
        x_poincare = hyperboloid.to_poincare(x, k=self.k)

        # Count points per cluster
        counts = torch.bincount(labels, minlength=self.n_clusters).float()
        active_mask = counts > 0

        # Initialize means with Euclidean mean (projected)
        mu_poincare = torch.zeros(self.n_clusters, dim, dtype=x.dtype, device=device)
        mu_poincare.index_add_(0, labels, x_poincare)

        # Avoid division by zero for empty clusters
        safe_counts = counts.clone()
        safe_counts[~active_mask] = 1.0
        mu_poincare = mu_poincare / safe_counts.unsqueeze(-1)
        mu_poincare = hyperboloid._poincare_project(mu_poincare, k=self.k)

        # Batched Frechet mean optimization
        # We only optimize non-empty clusters
        if active_mask.any():
            for _ in range(self.frechet_max_steps):
                # Gather current means for each point
                mu_expanded = mu_poincare[labels]

                # Compute gradient: log map from mu to each point
                log_x = hyperboloid._poincare_logmap(mu_expanded, x_poincare, k=self.k)

                # Aggregate gradients per cluster
                grad_sum = torch.zeros_like(mu_poincare)
                grad_sum.index_add_(0, labels, log_x)

                # Average gradient
                grad = grad_sum / safe_counts.unsqueeze(-1)

                # Zero out gradient for empty clusters to keep them stable (though we reinit them later)
                grad[~active_mask] = 0.0

                # Adaptive learning rate based on gradient norm
                grad_norm = grad.norm(dim=-1, p=2)

                # Check convergence (max gradient norm across all clusters)
                if grad_norm.max() < 1e-5:
                    break

                # Compute step size per cluster
                # lr = min(base_lr, 0.5 / (norm + 1e-8))
                current_lrs = torch.clamp(
                    0.5 / (grad_norm + 1e-8), max=self.frechet_lr
                ).unsqueeze(-1)

                delta_mu = current_lrs * grad

                # Update means using exponential map
                # Only update active clusters
                mu_poincare_active = mu_poincare[active_mask]
                delta_mu_active = delta_mu[active_mask]

                mu_new_active = hyperboloid._poincare_expmap(
                    mu_poincare_active, delta_mu_active, k=self.k
                )
                mu_poincare[active_mask] = hyperboloid._poincare_project(
                    mu_new_active, k=self.k
                )

        # Convert back to hyperboloid
        new_centroids = hyperboloid.from_poincare(mu_poincare, k=self.k)

        # Handle empty clusters: reinitialize to points with largest distance to current centroids
        if (~active_mask).any():
            empty_indices = torch.nonzero(~active_mask).squeeze(1)

            # Compute distances to currently active centroids
            active_centroids = new_centroids[active_mask]
            if len(active_centroids) > 0:
                # Find points that are furthest from any active centroid
                dists = self._compute_distances(x, active_centroids)
                min_dists, _ = torch.min(dists, dim=1)

                # Select top k furthest points for the k empty clusters
                # Use topk to find the best candidates
                _, furthest_indices = torch.topk(min_dists, len(empty_indices))
                new_centroids[empty_indices] = x[furthest_indices]
            else:
                # Fallback if all clusters somehow became empty (unlikely)
                rand_idx = torch.randint(
                    0, len(x), (len(empty_indices),), device=device
                )
                new_centroids[empty_indices] = x[rand_idx]

        return new_centroids

    def _init_centroids_random(
        self, x: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Initialize centroids by random selection.

        Args:
            x: torch.Tensor of shape (n, dim+1) - points on hyperboloid
            generator: torch random generator

        Returns:
            torch.Tensor of shape (n_clusters, dim+1) - initial centroids
        """
        n = x.shape[0]
        indices = torch.randperm(n, generator=generator)[: self.n_clusters]
        return x[indices].clone()

    def _init_centroids_kmeans_pp(
        self, x: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Initialize centroids using k-means++ algorithm.

        K-means++ selects initial centroids to be far apart, improving convergence.

        Args:
            x: torch.Tensor of shape (n, dim+1) - points on hyperboloid
            generator: torch random generator

        Returns:
            torch.Tensor of shape (n_clusters, dim+1) - initial centroids
        """
        n = x.shape[0]
        dim = x.shape[1]
        centroids = torch.zeros(self.n_clusters, dim, dtype=x.dtype, device=x.device)

        # Choose first centroid uniformly at random
        idx = torch.randint(0, n, (1,), generator=generator).item()
        centroids[0] = x[idx]

        # Keep track of minimum squared distance to any centroid
        # Initialize with infinity
        min_sq_distances = torch.full(
            (n,), float("inf"), dtype=x.dtype, device=x.device
        )

        # Choose remaining centroids with probability proportional to D(x)^2
        for i in range(1, self.n_clusters):
            # Compute distance from all points to the *newest* centroid (centroids[i-1])
            # x: (n, dim+1), centroids[i-1]: (dim+1) -> broadcast to (n, dim+1)
            dist_new = hyperboloid.distance(x, centroids[i - 1].unsqueeze(0), k=self.k)

            # Update minimum squared distances
            # min_sq_distances = min(old_min, distance_to_new_centroid^2)
            min_sq_distances = torch.minimum(min_sq_distances, dist_new**2)

            # Square distances for probability weights
            weights = min_sq_distances

            # Handle potential numerical instability if sum is 0 (should not happen with unique points)
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = torch.ones_like(weights) / n

            # Sample next centroid
            idx = torch.multinomial(weights, 1, generator=generator).item()
            centroids[i] = x[idx]

        return centroids

    def _single_fit(
        self, x: torch.Tensor, generator: torch.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
        """Run single K-means fitting.

        Args:
            x: torch.Tensor of shape (n, dim+1) - points on hyperboloid
            generator: torch random generator

        Returns:
            centroids: final centroids
            labels: cluster assignments
            inertia: sum of squared distances to centroids
            n_iter: number of iterations run
        """
        # Initialize centroids
        if self.init == "k-means++":
            centroids = self._init_centroids_kmeans_pp(x, generator)
        else:
            centroids = self._init_centroids_random(x, generator)

        labels = None
        prev_inertia = float("inf")

        for iteration in range(self.max_iter):
            # Assign clusters
            labels, distances = self._assign_clusters(x, centroids)
            inertia = (distances**2).sum().item()

            # Check convergence
            if abs(prev_inertia - inertia) < self.tol * abs(prev_inertia):
                break

            prev_inertia = inertia

            # Update centroids
            new_centroids = self._update_centroids(x, labels)

            # Check centroid movement
            centroid_shift = hyperboloid.distance(centroids, new_centroids, k=self.k)
            max_shift = centroid_shift.max().item()

            centroids = new_centroids

            if max_shift < self.tol:
                break

        # Final assignment
        labels, distances = self._assign_clusters(x, centroids)
        inertia = (distances**2).sum().item()

        return centroids, labels, inertia, iteration + 1

    def fit(self, x: Union[torch.Tensor, np.ndarray]) -> "LorentzKMeans":
        """Fit K-means clustering.

        Args:
            x: torch.Tensor or np.ndarray of shape (n, dim+1) - points on hyperboloid

        Returns:
            self
        """
        # Convert to torch if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Set up random generator
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)

        best_centroids = None
        best_labels = None
        best_inertia = float("inf")
        best_n_iter = 0

        for init_idx in range(self.n_init):
            if self.verbose:
                print(f"Initialization {init_idx + 1}/{self.n_init}")

            centroids, labels, inertia, n_iter = self._single_fit(x, generator)

            if self.verbose:
                print(f"  Inertia: {inertia:.4f}, Iterations: {n_iter}")

            if inertia < best_inertia:
                best_centroids = centroids
                best_labels = labels
                best_inertia = inertia
                best_n_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        if self.verbose:
            print(f"Best inertia: {best_inertia:.4f}")

        return self

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Predict cluster labels for new data.

        Args:
            x: torch.Tensor or np.ndarray of shape (n, dim+1) - points on hyperboloid

        Returns:
            torch.Tensor of shape (n,) - cluster labels
        """
        if self.cluster_centers_ is None:
            raise ValueError("Must call fit() before predict()")

        # Convert to torch if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        labels, _ = self._assign_clusters(x, self.cluster_centers_)
        return labels

    def fit_predict(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Fit and return cluster labels.

        Args:
            x: torch.Tensor or np.ndarray of shape (n, dim+1) - points on hyperboloid

        Returns:
            torch.Tensor of shape (n,) - cluster labels
        """
        self.fit(x)
        return self.labels_

    def transform(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Transform data to cluster-distance space.

        Args:
            x: torch.Tensor or np.ndarray of shape (n, dim+1) - points on hyperboloid

        Returns:
            torch.Tensor of shape (n, n_clusters) - distances to each centroid
        """
        if self.cluster_centers_ is None:
            raise ValueError("Must call fit() before transform()")

        # Convert to torch if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        return self._compute_distances(x, self.cluster_centers_)

    def fit_transform(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Fit and transform data to cluster-distance space.

        Args:
            x: torch.Tensor or np.ndarray of shape (n, dim+1) - points on hyperboloid

        Returns:
            torch.Tensor of shape (n, n_clusters) - distances to each centroid
        """
        self.fit(x)
        return self.transform(x)

    def __repr__(self):
        return (
            f"LorentzKMeans(n_clusters={self.n_clusters}, k={self.k}, "
            f"init='{self.init}', n_init={self.n_init})"
        )
