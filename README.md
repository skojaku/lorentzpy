# lorentzpy

A Python library for hyperbolic geometry operations on the Lorentz (hyperboloid) model with variable curvature support.

## Features

- **Coordinate conversions**: Convert between Lorentz hyperboloid and Poincare ball models
- **Frechet statistics**: Compute Frechet mean and variance on hyperbolic space
- **Data centering**: Center data by moving Frechet mean to the origin
- **HoroPCA**: Hyperbolic dimensionality reduction via horospherical projections
- **Lorentzian K-Means**: K-means clustering with Frechet mean centroids (scikit-learn compatible API)
- **Lorentzian KNN**: Efficient K-nearest neighbors search using FAISS
- **Monte Carlo sampling**: Sample points on the hyperboloid for density estimation
- **Visualization**: Plot density on the Poincare disk

## Installation

```bash
pip install lorentzpy
```

Or install from source:

```bash
git clone https://github.com/skojaku/lorentzpy.git
cd lorentzpy
pip install -e .
```

For GPU support with FAISS:

```bash
pip install lorentzpy[gpu]
```

## Quick Start

```python
import torch
import lorentzpy

# Create points in Poincare ball and convert to hyperboloid
poincare_points = torch.randn(100, 2) * 0.3  # 2D Poincare ball
hyperboloid_points = lorentzpy.from_poincare(poincare_points, k=1.0)

# Compute Frechet mean
mean = lorentzpy.frechet_mean(hyperboloid_points, k=1.0)

# Center the data
centered = lorentzpy.center(hyperboloid_points, k=1.0)

# Compute hyperbolic distances
dist = lorentzpy.distance(hyperboloid_points[0:1], hyperboloid_points[1:2], k=1.0)
```

## Conventions

- **Curvature parameter `k`**: `k > 0` represents the magnitude of negative curvature. Default is `k=1.0` for the unit hyperboloid.
- **Hyperboloid constraint**: Points satisfy `-x_0^2 + x_1^2 + ... + x_d^2 = -k`
- **Minkowski signature**: `(-1, +1, +1, ...)`
- **Coordinate format**: Hyperboloid points have shape `(batch, dim+1)` where the first coordinate is the time component.

---

## API Reference

### Coordinate Conversions

#### `lorentzpy.from_poincare(x, k=1.0, ideal=False)`

Convert from Poincare ball model to hyperboloid model.

**Parameters:**
- `x`: `torch.Tensor` of shape `(..., dim)` - Poincare ball coordinates
- `k`: `float` - Curvature parameter (default: 1.0)
- `ideal`: `bool` - True if input vectors are ideal points on the boundary

**Returns:**
- `torch.Tensor` of shape `(..., dim+1)` - Hyperboloid coordinates

**Example:**
```python
import torch
import lorentzpy

# 2D Poincare ball points
poincare = torch.tensor([[0.3, 0.2], [-0.1, 0.4]])
hyperboloid = lorentzpy.from_poincare(poincare, k=1.0)
print(hyperboloid.shape)  # torch.Size([2, 3])
```

---

#### `lorentzpy.to_poincare(x, k=1.0, ideal=False)`

Convert from hyperboloid model to Poincare ball model.

**Parameters:**
- `x`: `torch.Tensor` of shape `(..., dim+1)` - Hyperboloid coordinates
- `k`: `float` - Curvature parameter (default: 1.0)
- `ideal`: `bool` - True if input vectors are ideal points

**Returns:**
- `torch.Tensor` of shape `(..., dim)` - Poincare ball coordinates

**Example:**
```python
poincare_back = lorentzpy.to_poincare(hyperboloid, k=1.0)
```

---

#### `lorentzpy.distance(x, y, k=1.0)`

Compute hyperbolic distance on the hyperboloid.

**Formula:** `d(x, y) = sqrt(k) * arcosh(-<x, y>_L / k)`

**Parameters:**
- `x, y`: `torch.Tensor` of the same shape `(..., dim+1)` - Hyperboloid coordinates
- `k`: `float` - Curvature parameter (default: 1.0)

**Returns:**
- `torch.Tensor` of shape `(...)` - Hyperbolic distances

**Example:**
```python
x = lorentzpy.from_poincare(torch.tensor([[0.1, 0.2]]))
y = lorentzpy.from_poincare(torch.tensor([[0.5, 0.3]]))
d = lorentzpy.distance(x, y, k=1.0)
print(f"Distance: {d.item():.4f}")
```

---

#### `lorentzpy.expmap0(v, k=1.0, eps=1e-5)`

Exponential map at the origin of the hyperboloid.

Maps a tangent vector `v = (0, v_1, ..., v_d)` at the origin to a point on the hyperboloid.

**Parameters:**
- `v`: `torch.Tensor` of shape `(..., dim+1)` - Tangent vector at origin (first component should be 0)
- `k`: `float` - Curvature parameter (default: 1.0)
- `eps`: `float` - Numerical stability threshold

**Returns:**
- `torch.Tensor` of shape `(..., dim+1)` - Point on hyperboloid

**Example:**
```python
# Create tangent vector at origin
v = torch.tensor([[0.0, 0.5, 0.3]])
point = lorentzpy.expmap0(v, k=1.0)
```

---

#### `lorentzpy.logmap0(x, k=1.0, eps=1e-5)`

Logarithmic map at the origin of the hyperboloid.

Maps a point on the hyperboloid to a tangent vector at the origin.

**Parameters:**
- `x`: `torch.Tensor` of shape `(..., dim+1)` - Point on hyperboloid
- `k`: `float` - Curvature parameter (default: 1.0)
- `eps`: `float` - Numerical stability threshold

**Returns:**
- `torch.Tensor` of shape `(..., dim+1)` - Tangent vector at origin (first component is 0)

**Example:**
```python
# Map point back to tangent space
v_back = lorentzpy.logmap0(point, k=1.0)
```

---

#### `lorentzpy.update_time_coord(x, k=1.0, prepend_time_dim=False, eps=1e-5)`

Project spatial coordinates onto hyperboloid by computing the time coordinate.

Enforces the constraint: `x_0 = sqrt(k + ||x_spatial||^2)`

**Parameters:**
- `x`: `torch.Tensor` - Spatial coordinates (shape depends on `prepend_time_dim`)
- `k`: `float` - Curvature parameter (default: 1.0)
- `prepend_time_dim`: `bool` - If True, `x` contains only spatial coords `(..., dim)`. Otherwise `(..., dim+1)`.
- `eps`: `float` - Numerical stability threshold

**Returns:**
- `torch.Tensor` of shape `(..., dim+1)` - Points on hyperboloid

**Example:**
```python
# Project spatial coordinates onto hyperboloid
spatial = torch.tensor([[0.5, 0.3]])
on_hyperboloid = lorentzpy.update_time_coord(spatial, k=1.0, prepend_time_dim=True)
```

---

### Frechet Statistics

#### `lorentzpy.frechet_mean(x, k=1.0, lr=0.1, eps=1e-5, max_steps=5000, return_converged=False)`

Compute the Frechet mean of points on the hyperboloid.

The Frechet mean minimizes the sum of squared hyperbolic distances.

**Parameters:**
- `x`: `torch.Tensor` of shape `(n, dim+1)` - Hyperboloid coordinates
- `k`: `float` - Curvature parameter (default: 1.0)
- `lr`: `float` - Initial learning rate for gradient descent (default: 0.1)
- `eps`: `float` - Convergence threshold for gradient norm (default: 1e-5)
- `max_steps`: `int` - Maximum optimization steps (default: 5000)
- `return_converged`: `bool` - If True, also return convergence status

**Returns:**
- `mu`: `torch.Tensor` of shape `(1, dim+1)` - Frechet mean in hyperboloid coords
- `has_converged`: `bool` (only if `return_converged=True`)

**Example:**
```python
points = lorentzpy.from_poincare(torch.randn(100, 2) * 0.3)
mean = lorentzpy.frechet_mean(points, k=1.0)
print(f"Mean shape: {mean.shape}")  # torch.Size([1, 3])
```

---

#### `lorentzpy.frechet_variance(x, k=1.0, mu=None, lr=0.1, eps=1e-5, max_steps=5000, return_converged=False)`

Compute the Frechet variance of points on the hyperboloid.

Frechet variance is the mean squared distance from the Frechet mean.

**Parameters:**
- `x`: `torch.Tensor` of shape `(n, dim+1)` - Hyperboloid coordinates
- `k`: `float` - Curvature parameter (default: 1.0)
- `mu`: `torch.Tensor` of shape `(1, dim+1)` - Pre-computed Frechet mean (optional)
- `lr`, `eps`, `max_steps`: Parameters for mean computation if `mu` not provided
- `return_converged`: `bool` - If True, also return convergence status

**Returns:**
- `var`: `torch.Tensor` scalar - Frechet variance
- `has_converged`: `bool` (only if `return_converged=True`)

**Example:**
```python
var = lorentzpy.frechet_variance(points, k=1.0)
print(f"Variance: {var.item():.4f}")
```

---

### Centering

#### `lorentzpy.center(x, k=1.0, mu=None, lr=0.1, eps=1e-5, max_steps=5000, return_mean=False, return_converged=False)`

Center data to have Frechet mean at the origin.

Applies an isometry that moves the Frechet mean to the origin of the hyperboloid.

**Parameters:**
- `x`: `torch.Tensor` of shape `(n, dim+1)` - Hyperboloid coordinates
- `k`: `float` - Curvature parameter (default: 1.0)
- `mu`: `torch.Tensor` of shape `(1, dim+1)` - Pre-computed Frechet mean (optional)
- `lr`, `eps`, `max_steps`: Parameters for mean computation if `mu` not provided
- `return_mean`: `bool` - If True, also return the computed Frechet mean
- `return_converged`: `bool` - If True, also return convergence status

**Returns:**
- `x_centered`: `torch.Tensor` of shape `(n, dim+1)` - Centered hyperboloid coordinates
- `mu`: `torch.Tensor` (only if `return_mean=True`) - Frechet mean before centering
- `has_converged`: `bool` (only if `return_converged=True`)

**Example:**
```python
# Center data
centered = lorentzpy.center(points, k=1.0)

# Verify mean is at origin
new_mean = lorentzpy.frechet_mean(centered, k=1.0)
print(lorentzpy.to_poincare(new_mean))  # Should be close to [0, 0]
```

---

#### `lorentzpy.center_with_mean(x, mu, k=1.0)`

Center data given a pre-computed Frechet mean.

**Parameters:**
- `x`: `torch.Tensor` of shape `(n, dim+1)` - Hyperboloid coordinates
- `mu`: `torch.Tensor` of shape `(1, dim+1)` - Pre-computed Frechet mean
- `k`: `float` - Curvature parameter (default: 1.0)

**Returns:**
- `x_centered`: `torch.Tensor` of shape `(n, dim+1)` - Centered hyperboloid coordinates

---

### HoroPCA

#### `lorentzpy.HoroPCA`

Hyperbolic PCA using horospherical projections.

Based on: Chami et al., "HoroPCA: Hyperbolic Dimensionality Reduction via Horospherical Projections" (ICML 2021)

**Constructor:**
```python
HoroPCA(dim, n_components, k=1.0, lr=1e-2, max_steps=1000, frechet_variance=False, auc=False)
```

**Parameters:**
- `dim`: `int` - Input dimension (Poincare ball dimension = hyperboloid dimension - 1)
- `n_components`: `int` - Number of principal components to extract
- `k`: `float` - Curvature parameter (default: 1.0)
- `lr`: `float` - Learning rate for optimization (default: 1e-2)
- `max_steps`: `int` - Maximum optimization steps (default: 1000)
- `frechet_variance`: `bool` - If True, use Frechet variance per component
- `auc`: `bool` - If True, accumulate unexplained variance criterion

**Methods:**

##### `fit(x, tol=1e-5, patience=100, verbose=False, iterative=False)`

Fit HoroPCA model to data.

**Parameters:**
- `x`: `torch.Tensor` of shape `(batch_size, dim+1)` - Hyperboloid coordinates
- `tol`: `float` - Convergence tolerance for relative loss improvement
- `patience`: `int` - Steps without improvement before stopping
- `verbose`: `bool` - Print convergence info
- `iterative`: `bool` - Optimize components one at a time

**Note:** Data should be centered (Frechet mean at origin) for best results.

##### `transform(x)`

Transform data using fitted model.

**Returns:** `torch.Tensor` of shape `(batch_size, dim+1)` - Projected hyperboloid coordinates

##### `fit_transform(x, iterative=False)`

Fit model and transform data in one step.

##### `get_ideal_points()`

Get the ideal points (principal directions) in hyperboloid coordinates.

**Returns:** `torch.Tensor` of shape `(n_components, dim+1)`

##### `map_to_lower_dim(x)`

Map data to lower-dimensional hyperboloid.

**Returns:** `torch.Tensor` of shape `(batch_size, n_components+1)`

**Example:**
```python
# Create sample data
points = lorentzpy.from_poincare(torch.randn(500, 10) * 0.3)

# Center the data first
centered = lorentzpy.center(points, k=1.0)

# Fit HoroPCA
pca = lorentzpy.HoroPCA(dim=10, n_components=3, k=1.0)
pca.fit(centered, verbose=True)

# Transform to 3D
low_dim = pca.map_to_lower_dim(centered)
print(f"Reduced shape: {low_dim.shape}")  # torch.Size([500, 4])
```

---

### K-Means Clustering

#### `lorentzpy.LorentzKMeans`

K-Means clustering on the Lorentz hyperboloid using Frechet means as cluster centroids. API follows scikit-learn conventions.

**Algorithm:**
1. Initialize centroids using k-means++ (selects initial centroids far apart using hyperbolic distances) or random selection
2. Assign each point to nearest centroid using hyperbolic (geodesic) distance
3. Update centroids as Frechet mean of assigned points
4. Repeat until convergence

**Constructor:**
```python
LorentzKMeans(
    n_clusters,            # Number of clusters
    k=1.0,                 # Curvature parameter
    max_iter=100,          # Max iterations per initialization
    tol=1e-4,              # Convergence tolerance
    n_init=10,             # Number of initializations to try
    init="k-means++",      # "k-means++" or "random"
    random_state=None,     # Random seed for reproducibility
    frechet_lr=0.1,        # Learning rate for Frechet mean computation
    frechet_max_steps=100, # Max steps for Frechet mean
    verbose=False          # Print progress
)
```

**Attributes (after fitting):**
- `cluster_centers_`: `torch.Tensor` of shape `(n_clusters, dim+1)` - Cluster centroids on hyperboloid
- `labels_`: `torch.Tensor` of shape `(n_samples,)` - Cluster labels
- `inertia_`: `float` - Sum of squared distances to closest cluster center
- `n_iter_`: `int` - Number of iterations run

**Methods:**

##### `fit(x)`

Fit K-means clustering.

**Parameters:**
- `x`: `torch.Tensor` or `np.ndarray` of shape `(n, dim+1)` - Points on hyperboloid

**Returns:** `self`

##### `predict(x)`

Predict cluster labels for new data.

**Returns:** `torch.Tensor` of shape `(n,)` - Cluster labels

##### `fit_predict(x)`

Fit and return cluster labels.

**Returns:** `torch.Tensor` of shape `(n,)` - Cluster labels

##### `transform(x)`

Transform data to cluster-distance space.

**Returns:** `torch.Tensor` of shape `(n, n_clusters)` - Distances to each centroid

##### `fit_transform(x)`

Fit and transform data to cluster-distance space.

**Returns:** `torch.Tensor` of shape `(n, n_clusters)` - Distances to each centroid

**Example:**
```python
import torch
import lorentzpy

# Create embeddings on hyperboloid
poincare_pts = torch.randn(1000, 8) * 0.3
poincare_pts = poincare_pts / (1 + poincare_pts.norm(dim=-1, keepdim=True))
embeddings = lorentzpy.from_poincare(poincare_pts)

# Cluster
kmeans = lorentzpy.LorentzKMeans(n_clusters=10, random_state=42, verbose=True)
labels = kmeans.fit_predict(embeddings)

# Access results
print(f"Cluster sizes: {[(labels == i).sum().item() for i in range(10)]}")
print(f"Inertia: {kmeans.inertia_:.4f}")
print(f"Centroids shape: {kmeans.cluster_centers_.shape}")

# Verify centroids are on hyperboloid
for i, c in enumerate(kmeans.cluster_centers_[:3]):
    constraint = -c[0]**2 + (c[1:]**2).sum()
    print(f"Centroid {i} constraint (should be -1): {constraint.item():.6f}")

# Predict on new data
new_labels = kmeans.predict(new_embeddings)

# Transform to distance space
distances = kmeans.transform(embeddings)
```

---

### KNN Search

#### `lorentzpy.LorentzKNN`

K-Nearest Neighbors search using Lorentzian distance.

Uses FAISS inner product index for efficient computation.

**Constructor:**
```python
LorentzKNN(k=1.0)
```

**Parameters:**
- `k`: `float` - Curvature parameter (default: 1.0)

**Methods:**

##### `add(embeddings)`

Build FAISS index from embeddings.

**Parameters:**
- `embeddings`: `np.ndarray` or `torch.Tensor` of shape `(n, dim+1)` - Lorentz hyperboloid coordinates

**Returns:** `self` for method chaining

##### `search(embeddings, n_neighbors)`

Find k nearest neighbors.

**Parameters:**
- `embeddings`: `np.ndarray` or `torch.Tensor` of shape `(n, dim+1)` - Query points
- `n_neighbors`: `int` - Number of neighbors to find

**Returns:**
- `distances`: `np.ndarray` of shape `(n, n_neighbors)` - Hyperbolic distances
- `indices`: `np.ndarray` of shape `(n, n_neighbors)` - Neighbor indices

**Example:**
```python
# Create embeddings
embeddings = lorentzpy.from_poincare(torch.randn(1000, 8) * 0.3)

# Build index
knn = lorentzpy.LorentzKNN(k=1.0)
knn.add(embeddings)

# Query
query = embeddings[:10]
distances, indices = knn.search(query, n_neighbors=5)
print(f"Distances shape: {distances.shape}")  # (10, 5)
```

---

### Monte Carlo Sampling

#### `lorentzpy.monte_carlo_sample(embeddings, weights, n_samples, distance_sampler, k=1.0, max_distance=None, generator=None)`

Generate Monte Carlo samples for density estimation on hyperboloid.

**Algorithm:**
1. Sample center `mu` with probability proportional to weights
2. Sample uniform direction `u` on tangent space at `mu`
3. Sample distance `d` from the provided `distance_sampler`
4. Compute point via exponential map

**Parameters:**
- `embeddings`: `torch.Tensor` of shape `(N, d+1)` - Points on hyperboloid
- `weights`: `torch.Tensor` of shape `(N,)` - Sampling weights
- `n_samples`: `int` - Number of samples to generate
- `distance_sampler`: `Callable` - Function that takes `(n_samples, device, dtype, generator)` and returns distances
- `k`: `float` - Curvature parameter (default: 1.0)
- `max_distance`: `float` (optional) - Maximum distance to prevent overflow with heavy-tailed distributions
- `generator`: `torch.Generator` (optional) - Random generator

**Returns:**
- `torch.Tensor` of shape `(n_valid_samples, d+1)` - Sampled points on hyperboloid

**Example:**
```python
from lorentzpy import monte_carlo_sample, half_cauchy_sampler

# Create embeddings
embeddings = lorentzpy.from_poincare(torch.randn(100, 2) * 0.3)
weights = torch.ones(100)

# Sample with half-Cauchy distribution
sampler = half_cauchy_sampler(gamma=1.0)
samples = monte_carlo_sample(embeddings, weights, 1000, sampler, max_distance=10.0)
```

---

#### Distance Samplers

Built-in distance distribution samplers for use with `monte_carlo_sample`:

##### `lorentzpy.half_cauchy_sampler(gamma)`

Create a half-Cauchy distance sampler. Density: `p(d) ∝ 1/(d² + γ²)` for `d >= 0`

##### `lorentzpy.exponential_sampler(rate)`

Create an exponential distance sampler. Density: `p(d) = rate * exp(-rate * d)` for `d >= 0`

##### `lorentzpy.uniform_sampler(max_distance)`

Create a uniform distance sampler. Density: `p(d) = 1/max_distance` for `0 <= d <= max_distance`

##### `lorentzpy.gaussian_sampler(sigma)`

Create a half-Gaussian (folded normal) distance sampler. Samples `|X|` where `X ~ N(0, sigma^2)`

**Example:**
```python
# Different distribution samplers
cauchy = lorentzpy.half_cauchy_sampler(gamma=0.5)
exponential = lorentzpy.exponential_sampler(rate=2.0)
uniform = lorentzpy.uniform_sampler(max_distance=3.0)
gaussian = lorentzpy.gaussian_sampler(sigma=1.0)
```

---

#### `lorentzpy.compute_density_monte_carlo(...)`

Compute density on a 2D grid using Monte Carlo sampling.

**Parameters:**
- `embeddings`: `torch.Tensor` of shape `(N, 3)` - Embeddings on hyperboloid (2D case)
- `weights`: `torch.Tensor` - Weights for importance sampling
- `n_samples`: `int` - Number of Monte Carlo samples
- `distance_sampler`: `Callable` - Distance sampling function
- `k`: `float` - Curvature parameter
- `grid_size`: `int` - Output density grid resolution (default: 100)
- `xlim, ylim`: `Tuple[float, float]` (optional) - Grid limits in Poincare coordinates
- `max_distance`: `float` (optional) - Maximum distance for truncating samples
- `use_kde`: `bool` - If True, use KDE instead of histogram (default: False)
- `kde_bandwidth`: `float` (optional) - Bandwidth for KDE
- `generator`: `torch.Generator` (optional)

**Returns:**
- `X_grid, Y_grid`: Meshgrid arrays of shape `(grid_size, grid_size)`
- `density`: Density values of shape `(grid_size, grid_size)`

---

#### `lorentzpy.sample_tangent_direction(mu, k=1.0, generator=None)`

Sample uniformly distributed unit direction on tangent space at `mu`.

**Parameters:**
- `mu`: `torch.Tensor` of shape `(batch, d+1)` - Base points on hyperboloid
- `k`: `float` - Curvature parameter
- `generator`: `torch.Generator` (optional)

**Returns:**
- `torch.Tensor` of shape `(batch, d+1)` - Unit tangent vectors

---

#### `lorentzpy.project_to_tangent_space(mu, v, k=1.0)`

Project ambient vector `v` onto tangent space at `mu` on the hyperboloid.

**Note:** This is different from `logmap`. `logmap` maps a point ON the hyperboloid to the tangent space, while this function projects an arbitrary ambient vector onto the tangent space.

**Parameters:**
- `mu`: `torch.Tensor` of shape `(..., d+1)` - Base point on hyperboloid
- `v`: `torch.Tensor` of shape `(..., d+1)` - Ambient vector to project
- `k`: `float` - Curvature parameter

**Returns:**
- `torch.Tensor` of shape `(..., d+1)` - Projected tangent vector

---

### Visualization

#### `lorentzpy.plot_canvas(ax, max_distance=3.0, n_circles=10, n_angles=12)`

Add Poincare disk canvas with distance circles and angle lines.

**Parameters:**
- `ax`: matplotlib axes object
- `max_distance`: `float` - Maximum hyperbolic distance for circles (default: 3.0)
- `n_circles`: `int` - Number of equidistant circles (default: 10)
- `n_angles`: `int` - Number of angle lines (default: 12)

**Returns:**
- `ax`: matplotlib axes object

---

#### `lorentzpy.plot_density(ax, X_grid, Y_grid, densities, clip_to_disk=True, **contourf_kwargs)`

Plot density contours on Poincare disk.

**Parameters:**
- `ax`: matplotlib axes object
- `X_grid, Y_grid`: 2D arrays of coordinates from meshgrid
- `densities`: 2D array of density values
- `clip_to_disk`: `bool` - If True, clip to unit disk (default: True)
- `**contourf_kwargs`: Additional arguments for `contourf`

**Returns:**
- `contour`: matplotlib contour object

**Example:**
```python
import matplotlib.pyplot as plt
from lorentzpy import compute_density_monte_carlo, half_cauchy_sampler, plot_canvas, plot_density

# Compute density
X, Y, density = compute_density_monte_carlo(
    embeddings, weights, n_samples=10000,
    distance_sampler=half_cauchy_sampler(gamma=1.0),
    max_distance=5.0, use_kde=True
)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
plot_canvas(ax, max_distance=3.0)
plot_density(ax, X, Y, density, cmap='viridis')
plt.show()
```

---

### Projection Operations

#### `lorentzpy.orthogonal_projection(basis, x, k=1.0)`

Compute the orthogonal projection of `x` onto the geodesic submanifold.

The submanifold is the intersection of the hyperboloid with the Euclidean linear subspace spanned by the basis vectors.

**Parameters:**
- `basis`: `torch.Tensor` of shape `(num_basis, dim+1)` - Basis vectors
- `x`: `torch.Tensor` of shape `(batch_size, dim+1)` - Points to project
- `k`: `float` - Curvature parameter

**Returns:**
- `torch.Tensor` of shape `(batch_size, dim+1)` - Projected points

---

#### `lorentzpy.horo_projection(ideals, x, k=1.0)`

Compute the projection based on horosphere intersections.

**Parameters:**
- `ideals`: `torch.Tensor` of shape `(num_ideals, dim+1)` - Ideal points
- `x`: `torch.Tensor` of shape `(batch_size, dim+1)` - Points to project
- `k`: `float` - Curvature parameter

**Returns:**
- `proj_1, proj_2`: Two projections, each of shape `(batch_size, dim+1)`

---

#### `lorentzpy.exp_unit_tangents(base_points, unit_tangents, distances, k=1.0)`

Batched exponential map using given base points, unit tangent directions, and distances.

**Parameters:**
- `base_points`: `torch.Tensor` of shape `(..., dim+1)` - Points on hyperboloid
- `unit_tangents`: `torch.Tensor` of shape `(..., dim+1)` - Unit tangent vectors (Minkowski norm = 1, orthogonal to base point)
- `distances`: `torch.Tensor` of shape `(...)` - Distances to travel
- `k`: `float` - Curvature parameter

**Returns:**
- `torch.Tensor` of shape `(..., dim+1)` - New points on hyperboloid

---

### Minkowski Space Utilities

The `lorentzpy.minkowski` module provides low-level Minkowski space operations:

#### `lorentzpy.minkowski.bilinear_pairing(x, y)`

Compute the Minkowski bilinear pairing: `<x, y>_L = -x_0*y_0 + x_1*y_1 + x_2*y_2 + ...`

#### `lorentzpy.minkowski.squared_norm(x)`

Compute the squared Minkowski norm of `x`.

#### `lorentzpy.minkowski.pairwise_bilinear_pairing(x, y)`

Compute pairwise Minkowski bilinear pairings between batches of vectors.

#### `lorentzpy.minkowski.orthogonal_projection(basis, x)`

Compute orthogonal projection onto subspace (in Minkowski metric).

#### `lorentzpy.minkowski.reflection(subspace, x, subspace_given_by_normal=True)`

Compute reflection through a linear subspace (in Minkowski metric).

---

## Complete Example: Density Visualization

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import lorentzpy
from lorentzpy import (
    from_poincare, center, frechet_mean,
    monte_carlo_sample, half_cauchy_sampler,
    compute_density_monte_carlo, plot_canvas, plot_density
)

# Generate synthetic data: two clusters
torch.manual_seed(42)
cluster1 = torch.randn(50, 2) * 0.1 + torch.tensor([0.3, 0.2])
cluster2 = torch.randn(50, 2) * 0.1 + torch.tensor([-0.3, -0.1])
poincare_points = torch.cat([cluster1, cluster2], dim=0)

# Convert to hyperboloid
points = from_poincare(poincare_points, k=1.0)
weights = torch.ones(100)

# Compute density using Monte Carlo
X, Y, density = compute_density_monte_carlo(
    points, weights,
    n_samples=50000,
    distance_sampler=half_cauchy_sampler(gamma=0.5),
    k=1.0,
    grid_size=100,
    max_distance=5.0,
    use_kde=True,
    kde_bandwidth=0.05
)

# Visualize
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_facecolor('black')

# Plot density
plot_density(ax, X, Y, density, cmap='hot')

# Add canvas with distance circles
plot_canvas(ax, max_distance=3.0, n_circles=6)

# Overlay original points
poincare_np = poincare_points.numpy()
ax.scatter(poincare_np[:, 0], poincare_np[:, 1], c='cyan', s=20, alpha=0.7, zorder=5)

plt.title('Density Estimation on Poincare Disk')
plt.tight_layout()
plt.savefig('density_example.png', dpi=150, facecolor='black')
plt.show()
```

---

## License

MIT License

## Citation

If you use this library in your research, please cite:

```bibtex
@software{lorentzpy,
  author = {Sadamori Kojaku},
  title = {lorentzpy: Utilities for Lorentzian Embeddings},
  year = {2024},
  url = {https://github.com/skojaku/lorentzpy}
}
```
