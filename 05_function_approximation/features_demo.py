"""Part 1: what each feature construction looks like, and what its knobs do.

Reference
---------
Sutton and Barto, 2nd edition, chapter 9.  Polynomials 9.5.1, coarse coding
9.5.3, tile coding 9.5.4, radial basis functions 9.5.5, state aggregation 9.3.

The question the assignment asks is which hyperparameters each construction
has and how they change the shape of what it can represent.  The figures here
answer it by varying one knob at a time over a 2D state space and showing the
resulting feature map, which is more convincing than a table of names.

    construction        hyperparameters
    polynomial          degree
    RBF                 centres, sigma
    tile coding         centres, width, offsets (number of tilings)
    coarse coding       centres, width, offsets (number of fields)
    state aggregation   centres

Outputs
-------
figures/features_overview.png     one panel per construction at a fixed setting
figures/features_hyperparams.png  the effect of sigma, width and tilings
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit.features import (  # noqa: E402
    aggregation_features, coarse_features, make_centers, poly_features,
    rbf_features, tile_features,
)
from rlkit.plotting import save_fig  # noqa: E402

HERE = Path(__file__).resolve().parent
N_CENTERS = 100
EXTENT = [-0.2, 1.2, -0.2, 1.2]
OFFSETS = [(-0.1, 0.0), (0.0, 0.1), (0.1, 0.0), (0.0, -0.1)]


def grid_states(n: int = 100):
    """A regular grid of 2D states covering [-0.2, 1.2]^2."""
    axis = np.linspace(-0.2, 1.2, n)
    xx, yy = np.meshgrid(axis, axis)
    return np.stack([xx.ravel(), yy.ravel()], axis=1), n


def show(ax, activation, n, title):
    """Draw the total activation over the state grid."""
    ax.imshow(activation.reshape(n, n), extent=EXTENT, origin="lower", cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def overview(centers, states, n):
    """One panel per construction, at a single representative setting."""
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.2))
    panels = [
        ("polynomial, degree 2", poly_features(states, 2)),
        ("RBF, sigma 0.2", rbf_features(states, centers, 0.2)),
        ("tile coding, 1 tiling, width 0.2", tile_features(states, centers, 0.2)),
        ("tile coding, 4 tilings, width 0.2", tile_features(states, centers, 0.2, OFFSETS)),
        ("coarse coding, 4 fields, width 0.2", coarse_features(states, centers, 0.2, OFFSETS)),
        ("state aggregation", aggregation_features(states, centers)),
    ]
    for ax, (title, phi) in zip(axs.ravel(), panels):
        show(ax, phi.sum(axis=1), n, f"{title}\n{phi.shape[1]} features")
    fig.suptitle("Total feature activation over a 2D state space", fontsize=13)
    save_fig(fig, HERE / "figures" / "features_overview.png")


def hyperparameters(centers, states, n):
    """One row per knob, varying it while everything else is held fixed."""
    fig, axs = plt.subplots(4, 4, figsize=(13, 13))

    for c, degree in enumerate([1, 2, 3, 5]):
        phi = poly_features(states, degree)
        show(axs[0][c], phi.sum(axis=1), n, f"polynomial, degree {degree}\n{phi.shape[1]} features")

    for c, sigma in enumerate([0.05, 0.1, 0.3, 0.8]):
        show(axs[1][c], rbf_features(states, centers, sigma).sum(axis=1), n,
             f"RBF, sigma {sigma}")

    for c, width in enumerate([0.05, 0.1, 0.3, 0.8]):
        show(axs[2][c], tile_features(states, centers, width).sum(axis=1), n,
             f"tile, 1 tiling, width {width}")

    for c, k in enumerate([1, 2, 3, 4]):
        show(axs[3][c], tile_features(states, centers, 0.2, OFFSETS[:k]).sum(axis=1), n,
             f"tile, {k} tiling(s), width 0.2")

    fig.suptitle(
        "Each row varies one hyperparameter.  Polynomials gain global curvature with "
        "degree;\nwider sigmas and widths generalise more and resolve less; more tilings "
        "recover resolution\nwithout narrowing the tiles.",
        fontsize=11,
    )
    save_fig(fig, HERE / "figures" / "features_hyperparams.png")


def main() -> None:
    states, n = grid_states()
    centers = make_centers([-0.2, -0.2], [1.2, 1.2], [10, 10])
    print(f"  {centers.shape[0]} centres on a 10 x 10 grid over [-0.2, 1.2]^2")
    overview(centers, states, n)
    hyperparameters(centers, states, n)


if __name__ == "__main__":
    main()
