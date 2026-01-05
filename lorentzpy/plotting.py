"""Plotting utilities for visualizing Lorentzian embeddings.

Functions for visualizing embeddings in the Poincare disk representation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches


def plot_canvas(ax, max_distance=3.0, n_circles=10, n_angles=12):
    """Add Poincare disk canvas with distance circles and angle lines.

    Args:
        ax: matplotlib axes object
        max_distance: float, maximum hyperbolic distance for circles (default: 3.0)
        n_circles: int, number of equidistant circles (default: 10)
        n_angles: int, number of angle lines (default: 12)

    Returns:
        ax: matplotlib axes object
    """
    # Add circle boundary of Poincare disk
    circle = plt.Circle((0, 0), 1, fill=False, color="white", linestyle="--", alpha=0.5)
    ax.add_artist(circle)

    # Add radius circles at equal hyperbolic distances
    # Convert equidistant points in hyperbolic space to Poincare disk radii
    # r_poincare = tanh(d_hyperbolic / 2) for unit curvature
    lorentz_distances = np.linspace(0, max_distance, n_circles)
    radii = np.tanh(lorentz_distances / 2)

    for r in radii:
        circle = plt.Circle(
            (0, 0),
            r,
            fill=False,
            color="gray",
            alpha=0.5,
            linestyle="--",
            linewidth=0.5,
        )
        ax.add_artist(circle)

    # Add angle lines
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    max_radius = max(radii)
    for angle in angles:
        x = [0, max_radius * np.cos(angle)]
        y = [0, max_radius * np.sin(angle)]
        ax.plot(x, y, color="gray", alpha=0.4, linestyle=":", linewidth=0.5)

    return ax


def plot_density(ax, X_grid, Y_grid, densities, clip_to_disk=True, **contourf_kwargs):
    """Plot density contours on Poincare disk.

    Args:
        ax: matplotlib axes object
        X_grid: 2D array of x coordinates from meshgrid
        Y_grid: 2D array of y coordinates from meshgrid
        densities: 2D array of density values
        clip_to_disk: bool, if True, clip to unit disk (default: True)
        **contourf_kwargs: additional arguments for contourf

    Returns:
        contour: matplotlib contour object
    """
    # Convert to numpy if tensor
    if torch.is_tensor(densities):
        densities = densities.numpy()

    # Default contourf settings
    kwargs = {"levels": 100, "antialiased": True}
    kwargs.update(contourf_kwargs)

    # Create the contour plot
    contour = ax.contourf(X_grid, Y_grid, densities, **kwargs)

    if clip_to_disk:
        # Create a circular clip path
        clip_circle = patches.Circle((0, 0), 1, transform=ax.transData)
        contour.set_clip_path(clip_circle)

        # Add light overlay for better visibility
        circle = plt.Circle((0, 0), 1, color="white", alpha=0.1)
        ax.add_artist(circle)

    return contour
