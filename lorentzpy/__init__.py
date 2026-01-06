"""lorentzpy: Utilities for analyzing Lorentzian embeddings.

A package for hyperbolic geometry operations on the Lorentz hyperboloid model
with variable curvature support.

Main features:
- Coordinate conversions (Lorentz <-> Poincare ball)
- Frechet mean and variance computation
- Data centering
- HoroPCA dimensionality reduction
- Lorentzian KNN search
- Visualization (density plots on Poincare disk)

Convention:
    - Curvature parameter k > 0 represents magnitude of negative curvature
    - Hyperboloid constraint: -x_0^2 + x_1^2 + ... + x_d^2 = -k
    - Minkowski signature: (-1, +1, +1, ...)
    - Default k=1.0 for unit hyperboloid
"""

# Coordinate conversions and geometry
from .hyperboloid import (
    distance,
    to_poincare,
    from_poincare,
    expmap0,
    logmap0,
    update_time_coord,
    orthogonal_projection,
    horo_projection,
    exp_unit_tangents,
)

# Frechet statistics
from .frechet import frechet_mean, frechet_variance

# Centering
from .centering import center, center_with_mean

# HoroPCA
from .horopca import HoroPCA

# KNN
from .knn import LorentzKNN

# K-Means clustering
from .kmeans import LorentzKMeans

# Plotting
from .plotting import plot_canvas, plot_density

# Sampling
from .sampling import (
    monte_carlo_sample,
    compute_density_monte_carlo,
    half_cauchy_sampler,
    exponential_sampler,
    uniform_sampler,
    gaussian_sampler,
    student_t_sampler,
    sample_tangent_direction,
    project_to_tangent_space,
)

# Minkowski space utilities (less commonly needed directly)
from . import minkowski
from . import sampling

__all__ = [
    # Coordinate conversions
    "distance",
    "to_poincare",
    "from_poincare",
    "expmap0",
    "logmap0",
    "update_time_coord",
    "orthogonal_projection",
    "horo_projection",
    "exp_unit_tangents",
    # Frechet
    "frechet_mean",
    "frechet_variance",
    # Centering
    "center",
    "center_with_mean",
    # Classes
    "HoroPCA",
    "LorentzKNN",
    "LorentzKMeans",
    # Plotting
    "plot_canvas",
    "plot_density",
    # Sampling
    "monte_carlo_sample",
    "compute_density_monte_carlo",
    "half_cauchy_sampler",
    "exponential_sampler",
    "uniform_sampler",
    "gaussian_sampler",
    "student_t_sampler",
    "sample_tangent_direction",
    "project_to_tangent_space",
    # Modules
    "minkowski",
    "sampling",
]

__version__ = "0.1.0"
