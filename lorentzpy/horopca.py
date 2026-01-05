"""HoroPCA: Hyperbolic dimensionality reduction via horospherical projections.

Reference:
    Chami et al., "Horopca: Hyperbolic Dimensionality Reduction via
    Horospherical Projections" (ICML 2021)
"""

import torch
import torch.nn as nn

from . import hyperboloid
from . import _utils

MIN_NORM = 1e-15


# =============================================================================
# Horosphere projection utilities
# =============================================================================


def _busemann(x, p, k=1.0, keepdim=True):
    """Busemann function in Poincare ball.

    Args:
        x: (..., d) points in Poincare ball
        p: (..., d) ideal point direction
        k: float, curvature parameter (default: 1.0)
        keepdim: bool, keep last dimension

    Returns:
        (..., 1) if keepdim==True else (...)
    """
    import math
    sqrt_k = math.sqrt(k)

    xnorm = x.norm(dim=-1, p=2, keepdim=True)
    pnorm = p.norm(dim=-1, p=2, keepdim=True)
    p = p / pnorm.clamp_min(MIN_NORM)
    num = torch.norm(p - x, dim=-1, keepdim=True) ** 2
    den = (1 - xnorm**2).clamp_min(MIN_NORM)
    ans = torch.log((num / den).clamp_min(MIN_NORM)) / sqrt_k
    if not keepdim:
        ans = ans.squeeze(-1)
    return ans


def _circle_intersection_(r, R):
    """Compute intersection of circles with radii r and R, distance 1 between centers."""
    x = (1.0 - R**2 + r**2) / 2.0
    s = (r + R + 1) / 2.0
    sq_h = (s * (s - r) * (s - R) * (s - 1)).clamp_min(MIN_NORM)
    h = torch.sqrt(sq_h) * 2.0
    return x, h


def _busemann_to_horocycle(p, t, k=1.0):
    """Find the horocycle for level set of Busemann function to ideal point p with value t.

    Args:
        p: (..., d) ideal point direction
        t: (...) Busemann values
        k: float, curvature parameter (default: 1.0)

    Returns:
        c: (..., d) center of horocycle
        r: (...) radius of horocycle
    """
    import math
    sqrt_k = math.sqrt(k)
    q = -torch.tanh(sqrt_k * t / 2).unsqueeze(-1) * p
    c = (p + q) / 2.0
    r = torch.norm(p - q, dim=-1) / 2.0
    return c, r


def _sphere_intersection(c1, r1, c2, r2):
    """Compute intersection of spheres centered at ci with radius ri."""
    d = torch.norm(c1 - c2, dim=-1)
    x, h = _circle_intersection_(r1 / d.clamp_min(MIN_NORM), r2 / d.clamp_min(MIN_NORM))
    x = x.unsqueeze(-1)
    center = x * c2 + (1 - x) * c1
    radius = h * d
    return center, radius


def _sphere_intersections(c, r):
    """Compute intersection of k spheres in dimension d.

    Args:
        c: (..., k, d) list of centers
        r: (..., k) list of radii

    Returns:
        center: (..., d)
        radius: (...)
        ortho_directions: (..., d, k-1)
    """
    k = c.size(-2)
    assert k == r.size(-1)

    ortho_directions = []
    center = c[..., 0, :]
    radius = r[..., 0]
    for i in range(1, k):
        center, radius = _sphere_intersection(center, radius, c[..., i, :], r[..., i])
        ortho_directions.append(c[..., i, :] - center)
    ortho_directions.append(torch.zeros_like(center))
    ortho_directions = torch.stack(ortho_directions, dim=-1)
    return center, radius, ortho_directions


def _project_kd(p, x, k=1.0, keep_ambient=True):
    """Project n points in dimension d onto 'direction' spanned by k ideal points.

    Args:
        p: (..., k, d) ideal points
        x: (..., n, d) points to project
        k_curv: float, curvature parameter
        keep_ambient: bool, keep ambient dimension

    Returns:
        projection_1: (..., n, s) where s = d if keep_ambient else s = k
        projection_2: same as projection_1
        p: the ideal points
    """
    if len(p.shape) < 2:
        p = p.unsqueeze(0)
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    n_ideals = p.size(-2)
    d = x.size(-1)
    assert d == p.size(-1)

    busemann_distances = _busemann(x.unsqueeze(-2), p.unsqueeze(-3), k=k, keepdim=False)
    c, r = _busemann_to_horocycle(p.unsqueeze(-3), busemann_distances, k=k)
    c, r, ortho = _sphere_intersections(c, r)

    if ortho is None:
        direction = torch.ones_like(busemann_distances)
    else:
        a = torch.matmul(p.unsqueeze(-3), ortho)
        u, s, v = torch.linalg.svd(a, full_matrices=True)
        direction = u[..., -1]
    direction = direction @ p
    direction = direction / torch.norm(direction, dim=-1, keepdim=True).clamp_min(MIN_NORM)

    projection_1 = c - r.unsqueeze(-1) * direction
    projection_2 = c + r.unsqueeze(-1) * direction

    if not keep_ambient:
        _, _, v = torch.linalg.svd(p, full_matrices=True)
        projection_1 = (projection_1 @ v)[..., :n_ideals]
        projection_2 = (projection_2 @ v)[..., :n_ideals]
        p = (p @ v)[..., :n_ideals]

    return projection_1, projection_2, p


# =============================================================================
# HoroPCA class
# =============================================================================


class HoroPCA(nn.Module):
    """Hyperbolic PCA using horospherical projections.

    This implementation accepts hyperboloid coordinates as input and returns
    projections in hyperboloid coordinates.
    """

    def __init__(
        self,
        dim,
        n_components,
        k=1.0,
        lr=1e-2,
        max_steps=1000,
        frechet_variance=False,
        auc=False,
    ):
        """Initialize HoroPCA.

        Args:
            dim: int, input dimension (Poincare ball dimension, which is Minkowski_dim - 1)
            n_components: int, number of principal components to extract
            k: float, curvature parameter (default: 1.0)
            lr: float, learning rate for optimization (default: 1e-2)
            max_steps: int, maximum optimization steps (default: 1000)
            frechet_variance: bool, if True, use Frechet variance per component
            auc: bool, if True, accumulate unexplained variance criterion

        Note:
            auc=True and frechet_variance=True are not simultaneously supported.
        """
        super(HoroPCA, self).__init__()
        self.dim = dim
        self.n_components = n_components
        self.k = k
        self.lr = lr
        self.max_steps = max_steps
        self.frechet_variance = frechet_variance
        self.auc = auc

        # Initialize principal components (ideal point directions in Poincare ball)
        self.components = nn.ParameterList(
            nn.Parameter(torch.randn(1, dim)) for _ in range(n_components)
        )

        if self.frechet_variance:
            self.mean_weights = nn.Parameter(torch.zeros(n_components))

    def _get_components(self, orthogonalize=True):
        """Get component vectors, optionally orthogonalized.

        Args:
            orthogonalize: bool, if True, apply Gram-Schmidt

        Returns:
            torch.tensor of shape (n_components, dim)
        """
        Q = torch.cat([self.components[i] for i in range(self.n_components)])

        if orthogonalize:
            Q = self._gram_schmidt(Q)
        else:
            # Just normalize
            Q = Q / torch.norm(Q, dim=1, keepdim=True).clamp_min(MIN_NORM)

        return Q

    def _gram_schmidt(self, Q):
        """Apply Gram-Schmidt orthogonalization."""
        result = []
        for i in range(Q.shape[0]):
            v = Q[i]
            for u in result:
                v = v - torch.sum(u * v) / torch.sum(u * u).clamp_min(MIN_NORM) * u
            v = v / torch.norm(v).clamp_min(MIN_NORM)
            result.append(v)
        return torch.stack(result)

    def _orthonormal(self):
        """Get orthonormalized components via SVD."""
        Q = torch.cat([self.components[i].detach() for i in range(self.n_components)])
        Q = torch.nan_to_num(Q, nan=0.0)
        return _utils.orthonormal(Q)

    def _project(self, x_poincare, Q):
        """Project points onto submanifold spanned by ideal point directions Q.

        Args:
            x_poincare: torch.tensor of shape (batch_size, dim) - Poincare ball coords
            Q: torch.tensor of shape (n_components, dim) - ideal point directions

        Returns:
            Projected points in Poincare ball (batch_size, dim)
        """
        if self.n_components == 1:
            proj = _project_kd(Q, x_poincare, k=self.k)[0]
        else:
            # Use hyperboloid model for multi-component projection
            hyperboloid_ideals = hyperboloid.from_poincare(Q, k=self.k, ideal=True)
            hyperboloid_x = hyperboloid.from_poincare(x_poincare, k=self.k)
            hyperboloid_proj = hyperboloid.horo_projection(
                hyperboloid_ideals, hyperboloid_x, k=self.k
            )[0]
            proj = hyperboloid.to_poincare(hyperboloid_proj, k=self.k)
        return proj

    def _compute_variance(self, x_poincare, max_pairs=10000):
        """Compute variance of projected points.

        Args:
            x_poincare: torch.tensor of shape (batch_size, dim) - Poincare ball coords
            max_pairs: int, maximum pairs for variance estimation
        """
        if self.frechet_variance:
            Q = [
                self.mean_weights[i] * self.components[i]
                for i in range(self.n_components)
            ]
            mean = sum(Q).squeeze(0)
            distances = hyperboloid._poincare_distance(mean, x_poincare, k=self.k)
            var = torch.mean(distances**2)
        else:
            n = x_poincare.shape[0]
            if n * n <= max_pairs * 2:
                distances = hyperboloid._poincare_pairwise_distance(x_poincare, k=self.k)
                var = torch.mean(distances**2)
            else:
                idx1 = torch.randint(0, n, (max_pairs,), device=x_poincare.device)
                idx2 = torch.randint(0, n, (max_pairs,), device=x_poincare.device)
                mask = idx1 != idx2
                idx1, idx2 = idx1[mask], idx2[mask]
                distances = hyperboloid._poincare_distance(
                    x_poincare[idx1], x_poincare[idx2], k=self.k, keepdim=False
                )
                var = torch.mean(distances**2)
        return var

    def _compute_loss(self, x_poincare, Q):
        """Compute loss for optimization (negative variance)."""
        if self.n_components == 1:
            bus = _busemann(x_poincare, Q[0], k=self.k)
            return -torch.var(bus)
        else:
            if self.auc:
                auc = []
                for i in range(1, self.n_components):
                    Q_ = Q[:i, :]
                    proj = self._project(x_poincare, Q_)
                    var = self._compute_variance(proj)
                    auc.append(var)
                return -sum(auc)
            else:
                proj = self._project(x_poincare, Q)
                var = self._compute_variance(proj)
                return -var

    def fit(self, x, tol=1e-5, patience=100, verbose=False, iterative=False):
        """Fit HoroPCA model to data.

        Args:
            x: torch.tensor of shape (batch_size, dim+1) - hyperboloid coordinates
            tol: float, convergence tolerance for relative loss improvement
            patience: int, number of steps without improvement before stopping
            verbose: bool, if True, print convergence info
            iterative: bool, if True, optimize components iteratively

        Note:
            Data should be centered (Frechet mean at origin) for best results.
            Use lorentzpy.center() to center data before fitting.
        """
        # Convert hyperboloid to Poincare ball
        x_poincare = hyperboloid.to_poincare(x, k=self.k)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 3, min_lr=1e-6
        )

        loss_vals = []

        if not iterative:
            best_loss = float("inf")
            no_improve_count = 0

            if verbose:
                from tqdm import tqdm
                import sys
                iterator = tqdm(range(self.max_steps), file=sys.stdout)
            else:
                iterator = range(self.max_steps)

            for i in iterator:
                Q = self._get_components()
                loss = self._compute_loss(x_poincare, Q)
                loss_val = loss.item()
                loss_vals.append(loss_val)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1e5)
                optimizer.step()
                scheduler.step(loss_val)

                if verbose:
                    iterator.set_postfix({"loss": loss_val})

                if loss_val < best_loss - tol * abs(best_loss):
                    best_loss = loss_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        break
        else:
            # Iterative optimization: one component at a time
            for k_idx in range(self.n_components):
                component_params = [self.components[k_idx]]
                optimizer = torch.optim.Adam(component_params, lr=self.lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=patience // 3, min_lr=1e-6
                )
                no_improve_count = 0
                best_loss = float("inf")

                for i in range(self.max_steps):
                    Q = self._get_components()
                    loss = self._compute_loss(x_poincare, Q[:k_idx + 1, :])
                    loss_val = loss.item()
                    loss_vals.append(loss_val)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1e5)
                    optimizer.step()
                    scheduler.step(loss_val)

                    if loss_val < best_loss - tol * abs(best_loss):
                        best_loss = loss_val
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count >= patience:
                            break

                # Freeze this component
                self.components[k_idx].data = self._get_components()[k_idx].unsqueeze(0)
                self.components[k_idx].requires_grad = False

        if verbose:
            print(f"HoroPCA converged after {len(loss_vals)} steps")
            print(f"  Initial loss: {loss_vals[0]:.6f}")
            print(f"  Final loss:   {loss_vals[-1]:.6f}")

        return self

    def transform(self, x):
        """Transform data using fitted HoroPCA model.

        Args:
            x: torch.tensor of shape (batch_size, dim+1) - hyperboloid coordinates

        Returns:
            x_proj: torch.tensor of shape (batch_size, dim+1) - projected hyperboloid coords
        """
        x_poincare = hyperboloid.to_poincare(x, k=self.k)
        Q = self._get_components()
        x_proj_poincare = self._project(x_poincare, Q)
        return hyperboloid.from_poincare(x_proj_poincare, k=self.k)

    def fit_transform(self, x, iterative=False):
        """Fit model and transform data.

        Args:
            x: torch.tensor of shape (batch_size, dim+1) - hyperboloid coordinates
            iterative: bool, if True, optimize components iteratively

        Returns:
            x_proj: torch.tensor of shape (batch_size, dim+1) - projected hyperboloid coords
        """
        self.fit(x, iterative=iterative)
        return self.transform(x)

    def get_ideal_points(self):
        """Get the ideal points (principal directions) in hyperboloid coordinates.

        Returns:
            torch.tensor of shape (n_components, dim+1) - ideal points in hyperboloid
        """
        Q = self._get_components()
        return hyperboloid.from_poincare(Q, k=self.k, ideal=True)

    def map_to_lower_dim(self, x):
        """Map data to lower-dimensional hyperboloid.

        Args:
            x: torch.tensor of shape (batch_size, dim+1) - hyperboloid coordinates

        Returns:
            torch.tensor of shape (batch_size, n_components+1) - lower-dim hyperboloid coords
        """
        x_poincare = hyperboloid.to_poincare(x, k=self.k)
        Q = self._get_components()
        x_proj_poincare = self._project(x_poincare, Q)

        # Project to lower-dimensional subspace
        Q_orthonormal = self._orthonormal()
        x_low_poincare = x_proj_poincare @ Q_orthonormal.T

        return hyperboloid.from_poincare(x_low_poincare, k=self.k)
