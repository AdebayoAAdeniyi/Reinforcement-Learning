"""Linear function approximation features.

Five feature constructions from Sutton and Barto chapter 9:

==================  ==========================  ==================================
function            book section                hyperparameters
==================  ==========================  ==================================
`poly_features`     9.5.1 polynomials           degree
`rbf_features`      9.5.5 radial basis          centers, sigmas
`tile_features`     9.5.4 tile coding           centers, widths, offsets
`coarse_features`   9.5.3 coarse coding         centers, widths, offsets
`aggregation_...`   9.3  state aggregation      centers
==================  ==========================  ==================================

Every function takes states of shape (N, S) and returns features of shape
(N, D), including when N is 1.  Keeping that contract is what lets the same
feature function be used for batch regression, for a single step of TD, and
for the policy gradient chapters without reshaping at each call site.

Tile coding and coarse coding differ in exactly one respect.  Tile coding
produces a binary indicator: a state either falls inside a tile or it does
not.  Coarse coding keeps a graded value inside each field, so nearby states
inside the same field are distinguishable.  That difference is why coarse
coding fits the gridworld value function better here, and it is worth being
able to see it in the code rather than taking it on trust.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "poly_features",
    "rbf_features",
    "tile_features",
    "coarse_features",
    "aggregation_features",
    "make_centers",
]


def poly_features(state: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial basis up to `degree`, including the bias term.

    For a one dimensional state this gives [1, s, s^2, ..., s^degree].
    For an S dimensional state it gives every monomial whose total degree is
    at most `degree`, which is the construction in equation 9.17, not the
    per dimension powers that a naive implementation produces.
    """
    from itertools import combinations_with_replacement

    state = np.atleast_2d(state)
    n, s_dim = state.shape
    cols = [np.ones(n)]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(s_dim), d):
            cols.append(np.prod(state[:, combo], axis=1))
    return np.stack(cols, axis=1)


def rbf_features(state: np.ndarray, centers: np.ndarray, sigmas) -> np.ndarray:
    """Gaussian radial basis functions, equation 9.8.

    phi_i(s) = exp( - ||s - c_i||^2 / (2 sigma_i^2) ), evaluated per dimension
    so that `sigmas` may be a scalar or one width per state dimension.
    """
    state = np.atleast_2d(state)
    z = (state[:, None, :] - centers[None, :, :]) / np.asarray(sigmas)
    return np.exp(-0.5 * (z ** 2).sum(axis=-1))


def tile_features(
    state: np.ndarray,
    centers: np.ndarray,
    widths,
    offsets: list[tuple[float, ...]] | None = None,
) -> np.ndarray:
    """Binary tile coding, section 9.5.4.

    One tiling per entry of `offsets`.  A feature is 1 when the state lies
    inside the axis aligned box of half width `widths` around the offset
    centre, and 0 otherwise.  With several tilings the features of each tiling
    are summed, so a state that falls inside the same tile index in k tilings
    gets the value k.
    """
    state = np.atleast_2d(state)
    offsets = offsets if offsets is not None else [tuple([0.0] * state.shape[1])]
    widths = np.asarray(widths, dtype=float)
    out = np.zeros((state.shape[0], centers.shape[0]))
    for off in offsets:
        shifted = centers + np.asarray(off, dtype=float)
        inside = np.abs(state[:, None, :] - shifted[None, :, :]) <= widths / 2.0
        out += inside.all(axis=-1)
    return out


def coarse_features(
    state: np.ndarray,
    centers: np.ndarray,
    widths,
    offsets: list[tuple[float, ...]] | None = None,
) -> np.ndarray:
    """Coarse coding with graded receptive fields, section 9.5.3.

    Same geometry as `tile_features`, but instead of a 0/1 indicator each
    active field returns `1 - ||s - c|| / radius`, so the activation falls off
    linearly from the centre of the field to its edge and is 0 outside.  That
    graded response is the only difference from tile coding, and it is what
    lets a state be located within a field rather than only within a tile.
    """
    state = np.atleast_2d(state)
    offsets = offsets if offsets is not None else [tuple([0.0] * state.shape[1])]
    widths = np.asarray(widths, dtype=float)
    radius = float(np.mean(widths)) / 2.0
    out = np.zeros((state.shape[0], centers.shape[0]))
    for off in offsets:
        shifted = centers + np.asarray(off, dtype=float)
        delta = state[:, None, :] - shifted[None, :, :]
        inside = (np.abs(delta) <= widths / 2.0).all(axis=-1)
        dist = np.sqrt((delta ** 2).sum(axis=-1))
        out += inside * np.clip(1.0 - dist / (radius + 1e-12), 0.0, 1.0)
    return out


def aggregation_features(state: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """State aggregation, section 9.3.

    A one hot vector marking the nearest centre.  This is the special case of
    coarse coding in which the fields do not overlap, so there is no
    generalisation between groups at all.
    """
    state = np.atleast_2d(state)
    dist = ((state[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return (dist == dist.min(axis=-1, keepdims=True)) * 1.0


def make_centers(low, high, n_per_dim, pad: float = 0.0) -> np.ndarray:
    """Evenly spaced centres on a grid covering [low, high] in every dimension.

    `pad` widens the range by that fraction of a cell on each side, which
    stops the outermost centres from sitting exactly on the boundary where
    half of their receptive field would fall outside the state space.
    """
    low, high = np.atleast_1d(low).astype(float), np.atleast_1d(high).astype(float)
    n_per_dim = np.atleast_1d(n_per_dim).astype(int)
    axes = [
        np.linspace(
            low[i] - (high[i] - low[i]) / n_per_dim[i] * pad,
            high[i] + (high[i] - low[i]) / n_per_dim[i] * pad,
            n_per_dim[i],
        )
        for i in range(len(low))
    ]
    return np.array(np.meshgrid(*axes)).reshape(len(low), -1).T
