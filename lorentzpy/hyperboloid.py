"""Hyperboloid model operations with curvature support.

Convention: The ambient Minkowski space has signature (-1, 1, 1, ...).
Points on the hyperboloid satisfy: -x_0^2 + x_1^2 + ... + x_d^2 = -k
where k > 0 is the curvature parameter (magnitude of negative curvature).

We use the positive sheet, i.e., every point has positive first coordinate (x_0 > 0).
"""

import torch
import math

from . import minkowski

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


# =============================================================================
# Core hyperboloid operations
# =============================================================================


def distance(x, y, k=1.0):
    """Compute hyperbolic distance on the hyperboloid.

    Formula: d(x, y) = sqrt(k) * arcosh(-<x, y>_L / k)

    Args:
        x, y: torch.tensor of the same shape (..., dim+1) - hyperboloid coordinates
        k: float, curvature parameter (default: 1.0)

    Returns:
        torch.tensor of shape (...)
    """
    inner = minkowski.bilinear_pairing(x, y)
    return math.sqrt(k) * torch.acosh(torch.clamp(-inner / k, min=1.0))


def exp_unit_tangents(base_points, unit_tangents, distances, k=1.0):
    """Batched exponential map using given base points, unit tangent directions, and distances.

    Args:
        base_points: torch.tensor of shape (..., dim+1) - points on hyperboloid
        unit_tangents: torch.tensor of shape (..., dim+1) - unit tangent vectors
            Each must have Minkowski squared norm 1 and be orthogonal to base_point
        distances: torch.tensor of shape (...) - distances to travel
        k: float, curvature parameter (default: 1.0)

    Returns:
        torch.tensor of shape (..., dim+1) - new points on hyperboloid
    """
    sqrt_k = math.sqrt(k)
    distances = distances.unsqueeze(-1) / sqrt_k
    return base_points * torch.cosh(distances) + unit_tangents * sqrt_k * torch.sinh(distances)


def from_poincare(x, k=1.0, ideal=False):
    """Convert from Poincare ball model to hyperboloid model.

    Args:
        x: torch.tensor of shape (..., dim) - Poincare ball coordinates
        k: float, curvature parameter (default: 1.0)
        ideal: bool, True if input vectors are ideal points (on boundary)

    Returns:
        torch.tensor of shape (..., dim+1) - hyperboloid coordinates
    """
    sqrt_k = math.sqrt(k)
    if ideal:
        t = sqrt_k * torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype).unsqueeze(-1)
        return torch.cat((t, x), dim=-1)
    else:
        eucl_squared_norm = (x * x).sum(dim=-1, keepdim=True)
        denom = (1 - eucl_squared_norm).clamp_min(MIN_NORM)
        return sqrt_k * torch.cat((1 + eucl_squared_norm, 2 * x), dim=-1) / denom


def to_poincare(x, k=1.0, ideal=False):
    """Convert from hyperboloid model to Poincare ball model.

    Args:
        x: torch.tensor of shape (..., dim+1) - hyperboloid coordinates
        k: float, curvature parameter (default: 1.0)
        ideal: bool, True if input vectors are ideal points

    Returns:
        torch.tensor of shape (..., dim) - Poincare ball coordinates
    """
    sqrt_k = math.sqrt(k)
    if ideal:
        return x[..., 1:] / (x[..., 0] / sqrt_k).unsqueeze(-1).clamp_min(MIN_NORM)
    else:
        return x[..., 1:] / (x[..., 0] + sqrt_k).unsqueeze(-1).clamp_min(MIN_NORM)


def update_time_coord(x, k=1.0, prepend_time_dim=False, eps=1e-5):
    """Project spatial coordinates onto hyperboloid by computing the time coordinate.

    Enforces the constraint: -x_0^2 + ||x_spatial||^2 = -k
    => x_0 = sqrt(k + ||x_spatial||^2)

    Args:
        x: torch.tensor of shape (..., dim) if prepend_time_dim=True, else (..., dim+1)
        k: float, curvature parameter (default: 1.0)
        prepend_time_dim: bool, if True, x contains only spatial coords
        eps: float, numerical stability threshold

    Returns:
        torch.tensor of shape (..., dim+1) - points on hyperboloid
    """
    if prepend_time_dim:
        spatial = x
    else:
        spatial = x[..., 1:]

    spatial_norm_sq = torch.sum(spatial * spatial, dim=-1, keepdim=True)
    x0 = torch.sqrt(torch.clamp(k + spatial_norm_sq, min=eps))
    return torch.cat((x0, spatial), dim=-1)


def orthogonal_projection(basis, x, k=1.0):
    """Compute the orthogonal projection of x onto the geodesic submanifold.

    The submanifold is the intersection of the hyperboloid with the Euclidean
    linear subspace spanned by the basis vectors.

    Args:
        basis: torch.tensor of shape (num_basis, dim+1)
        x: torch.tensor of shape (batch_size, dim+1)
        k: float, curvature parameter (default: 1.0)

    Returns:
        torch.tensor of shape (batch_size, dim+1)

    Conditions:
        - Each basis vector must have non-positive Minkowski squared norms.
        - There must be at least 2 basis vectors.
        - The basis vectors must be linearly independent.
    """
    minkowski_proj = minkowski.orthogonal_projection(basis, x)
    squared_norms = minkowski.squared_norm(minkowski_proj)
    # Project onto hyperboloid with curvature k
    return minkowski_proj / torch.sqrt(-squared_norms / k).unsqueeze(-1)


def horo_projection(ideals, x, k=1.0):
    """Compute the projection based on horosphere intersections.

    The target submanifold has dimension num_ideals and is a geodesic submanifold
    passing through the ideal points and the origin (sqrt(k), 0, 0, ...).

    Args:
        ideals: torch.tensor of shape (num_ideals, dim+1) - ideal points
            num_ideals must be STRICTLY between 1 and dim+1
        x: torch.tensor of shape (batch_size, dim+1) - points to project
        k: float, curvature parameter (default: 1.0)

    Returns:
        proj_1, proj_2: Two projections, torch.tensor of shape (batch_size, dim+1)
    """
    sqrt_k = math.sqrt(k)

    # Compute orthogonal (geodesic) projection from x to the geodesic submanifold spanned by ideals
    spine_ortho_proj = orthogonal_projection(ideals, x, k)
    spine_dist = distance(spine_ortho_proj, x, k)

    # origin on hyperboloid with curvature k: [sqrt(k), 0, 0, ...]
    origin = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
    origin[0] = sqrt_k

    # Find a tangent vector of the hyperboloid at spine_ortho_proj that is tangent
    # to the target submanifold and orthogonal to the spine.
    chords = origin - spine_ortho_proj
    tangents = chords - minkowski.orthogonal_projection(ideals, chords)
    unit_tangents = tangents / torch.sqrt(minkowski.squared_norm(tangents)).view(-1, 1)

    proj_1 = exp_unit_tangents(spine_ortho_proj, unit_tangents, spine_dist, k)
    proj_2 = exp_unit_tangents(spine_ortho_proj, unit_tangents, -spine_dist, k)

    return proj_1, proj_2


# =============================================================================
# Exponential and logarithmic maps
# =============================================================================


def expmap0(v, k=1.0, eps=1e-5):
    """Exponential map at the origin of the hyperboloid.

    Maps tangent vector v = (0, v_1, ..., v_d) to point on hyperboloid.

    Args:
        v: torch.tensor of shape (..., dim+1) - tangent vector at origin
           The first component should be 0 (will be ignored)
        k: float, curvature parameter (default: 1.0)
        eps: float, numerical stability threshold

    Returns:
        torch.tensor of shape (..., dim+1) - point on hyperboloid
    """
    sqrt_k = math.sqrt(k)
    v_spatial = v[..., 1:]
    v_norm = torch.sqrt(torch.clamp(torch.sum(v_spatial * v_spatial, dim=-1, keepdim=True), min=eps))
    v_norm_scaled = v_norm / sqrt_k

    x0 = sqrt_k * torch.cosh(v_norm_scaled)

    # sinh(x)/x -> 1 as x -> 0, handle small norms
    scale = torch.where(
        v_norm > eps,
        sqrt_k * torch.sinh(v_norm_scaled) / v_norm,
        torch.ones_like(v_norm),
    )
    x_spatial = scale * v_spatial

    return torch.cat((x0, x_spatial), dim=-1)


def logmap0(x, k=1.0, eps=1e-5):
    """Logarithmic map at the origin of the hyperboloid.

    Maps point on hyperboloid to tangent vector at origin.

    Args:
        x: torch.tensor of shape (..., dim+1) - point on hyperboloid
        k: float, curvature parameter (default: 1.0)
        eps: float, numerical stability threshold

    Returns:
        torch.tensor of shape (..., dim+1) - tangent vector at origin
           First component is 0
    """
    sqrt_k = math.sqrt(k)
    x0 = x[..., :1]
    x_spatial = x[..., 1:]

    # Distance from origin to x
    d = sqrt_k * torch.acosh(torch.clamp(x0 / sqrt_k, min=1.0 + eps))

    x_spatial_norm = torch.sqrt(torch.clamp(torch.sum(x_spatial * x_spatial, dim=-1, keepdim=True), min=eps))

    # Handle points at origin
    scale = torch.where(
        x_spatial_norm > eps,
        d / x_spatial_norm,
        torch.ones_like(x_spatial_norm),
    )

    v_spatial = scale * x_spatial
    v0 = torch.zeros_like(x0)

    return torch.cat((v0, v_spatial), dim=-1)


# =============================================================================
# Poincare ball operations (used internally for optimization)
# =============================================================================


def _poincare_project(x, k=1.0):
    """Project points to Poincare ball boundary."""
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS.get(x.dtype, 1e-5)
    maxnorm = 1 - eps
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def _poincare_expmap0(u, k=1.0):
    """Exponential map at the origin of the Poincare ball."""
    sqrt_k = math.sqrt(k)
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    # Scale by curvature
    gamma_1 = torch.tanh(sqrt_k * u_norm) * u / (sqrt_k * u_norm)
    return _poincare_project(gamma_1, k)


def _poincare_logmap0(y, k=1.0):
    """Logarithmic map at the origin of the Poincare ball."""
    sqrt_k = math.sqrt(k)
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm * torch.atanh(y_norm.clamp(-1 + 1e-15, 1 - 1e-15)) / sqrt_k


def _poincare_lambda(x, k=1.0):
    """Compute the conformal factor (lambda_x)."""
    x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
    return 2 / (1.0 - x_sqnorm).clamp_min(MIN_NORM)


def _poincare_mobius_add(x, y, k=1.0):
    """Mobius addition in Poincare ball."""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def _poincare_mobius_mul(x, t, k=1.0):
    """Mobius scalar multiplication in Poincare ball."""
    sqrt_k = math.sqrt(k)
    normx = x.norm(dim=-1, p=2, keepdim=True).clamp(min=MIN_NORM, max=1.0 - 1e-5)
    return torch.tanh(t * torch.atanh(normx)) * x / normx


def _poincare_expmap(x, u, k=1.0):
    """Exponential map at arbitrary point x in Poincare ball."""
    sqrt_k = math.sqrt(k)
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = torch.tanh(_poincare_lambda(x, k) * sqrt_k * u_norm / 2) * u / (sqrt_k * u_norm)
    gamma_1 = _poincare_mobius_add(x, second_term, k)
    return gamma_1


def _poincare_logmap(x, y, k=1.0):
    """Logarithmic map at arbitrary point x in Poincare ball."""
    sqrt_k = math.sqrt(k)
    sub = _poincare_mobius_add(-x, y, k)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM).clamp_max(1 - 1e-15)
    return 2 / (_poincare_lambda(x, k) * sqrt_k) * torch.atanh(sub_norm) * sub / sub_norm


def _poincare_distance(x, y, k=1.0, keepdim=True):
    """Hyperbolic distance on the Poincare ball."""
    sqrt_k = math.sqrt(k)
    pairwise_norm = _poincare_mobius_add(-x, y, k).norm(dim=-1, p=2, keepdim=True)
    dist = 2.0 / sqrt_k * torch.atanh(pairwise_norm.clamp(-1 + MIN_NORM, 1 - MIN_NORM))
    if not keepdim:
        dist = dist.squeeze(-1)
    return dist


def _poincare_pairwise_distance(x, k=1.0, keepdim=False):
    """All pairs of hyperbolic distances (NxN matrix)."""
    return _poincare_distance(x.unsqueeze(-2), x.unsqueeze(-3), k, keepdim=keepdim)


def _poincare_distance0(x, k=1.0, keepdim=True):
    """Compute hyperbolic distance between x and the origin."""
    sqrt_k = math.sqrt(k)
    x_norm = x.norm(dim=-1, p=2, keepdim=True)
    d = 2 / sqrt_k * torch.atanh(x_norm.clamp(-1 + 1e-15, 1 - 1e-15))
    if not keepdim:
        d = d.squeeze(-1)
    return d


def _poincare_midpoint(x, y, k=1.0):
    """Compute hyperbolic midpoint between x and y in Poincare ball."""
    t1 = _poincare_mobius_add(-x, y, k)
    t2 = _poincare_mobius_mul(t1, 0.5, k)
    return _poincare_mobius_add(x, t2, k)


def _poincare_orthogonal_projection(x, Q, k=1.0, normalized=False):
    """Orthogonally project x onto linear subspace spanned by rows of Q in Poincare ball."""
    from . import _utils
    if not normalized:
        Q = _utils.orthonormal(Q)
    x_ = _utils.reflect(x, Q)
    return _poincare_midpoint(x, x_, k)
