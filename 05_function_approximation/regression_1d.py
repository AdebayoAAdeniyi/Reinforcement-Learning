"""Part 2: fitting a 1D target with each of the five feature constructions.

This is supervised regression, not reinforcement learning, and that is the
point.  Before asking a linear approximator to chase a bootstrapped target it
is worth seeing what it can and cannot represent when the target is fixed and
known, because any failure here will also be a failure inside TD, only harder
to diagnose.

Reference
---------
Sutton and Barto, 2nd edition, section 9.4 (linear methods) and 9.5
(feature construction).  The update is plain gradient descent on the mean
squared error, equation 9.5 with the true target in place of U_t.

Targets
-------
`smooth`    sin(x) + 0.1 x^2 on [-10, 10].  Continuous and gently curved, so
            a low order polynomial can almost represent it.
`piecewise` a deliberately hostile target, pieced together out of cubics, an
            exponential, two flat sections and a cosine burst, with two jump
            discontinuities.  No polynomial of modest degree can represent
            it at all.

The pair is chosen to separate "the optimiser needs more steps" from "the
representation cannot express the answer", which are routinely confused when
a fit looks bad.

Outputs
-------
figures/regression_smooth.png
figures/regression_piecewise.png
results/regression_mse.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit.features import (  # noqa: E402
    aggregation_features, coarse_features, poly_features, rbf_features, tile_features,
)
from rlkit.plotting import save_fig  # noqa: E402

HERE = Path(__file__).resolve().parent
X = np.linspace(-10, 10, 100)
MAX_ITER = 20_000
THRESH = 1e-10


def target_smooth() -> np.ndarray:
    return np.sin(X) + 0.1 * X ** 2


def target_piecewise() -> np.ndarray:
    """A deliberately hostile target: two jump discontinuities and a burst."""
    y = np.zeros(X.shape)
    y[0:10] = X[0:10] ** 3 / 3.0
    y[10:20] = np.exp(X[25:35])
    y[20:30] = -X[0:10] ** 3 / 2.0
    y[30:60] = 100.0
    y[60:70] = 0.0
    y[70:100] = np.cos(X[70:100]) * 100.0
    return y


def fit(phi: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Batch gradient descent on the mean squared error.

    Returns the prediction, the MSE history and the number of iterations.
    The step size is scaled by the largest eigenvalue of the feature Gram
    matrix.  Without that, a single `alpha` cannot work for all five
    constructions at once: a polynomial basis on [-10, 10] has features of
    magnitude 10^3 while tile coding has features of magnitude 1, so a step
    size that converges for one diverges for the other.  Hand tuning `alpha`
    per method instead would make the MSE comparison between methods partly a
    comparison of how carefully each step size was tuned.
    """
    n, d = phi.shape
    scale = np.linalg.eigvalsh(phi.T @ phi / n).max()
    step = alpha / max(scale, 1e-12)
    w = np.zeros(d)
    history = []
    for it in range(MAX_ITER):
        pred = phi @ w
        err = y - pred
        mse = float(np.mean(err ** 2))
        history.append(mse)
        w = w + step * (phi.T @ err) / n
        if it > 0 and abs(history[-2] - mse) < THRESH:
            break
    return phi @ w, np.array(history), len(history)


def build_features(n_centers: int):
    """The five constructions, all with a comparable number of features.

    `n_centers` is deliberately a parameter, and the script runs the sweep
    at both 20 and 100 centres for 100 data points.  At 100 the three local
    constructions have one free parameter per sample, so each interpolates
    the data exactly and scores a mean squared error of zero on any target
    whatsoever; any ranking between them there is noise.  Fewer centres than
    samples is what actually tests generalisation.  Showing both is the
    point: a comparison can be set up so that it cannot answer its own
    question, and the symptom is a suspiciously perfect score.
    """
    centers = np.linspace(-10, 10, n_centers).reshape(-1, 1)
    width = 20.0 / n_centers * 2.0        # two cell widths, so fields overlap
    sigma = 20.0 / n_centers * 1.5
    offsets = [(-width / 4,), (0.0,), (width / 4,)]
    xs = X.reshape(-1, 1)
    return {
        "polynomial (deg 5)": poly_features(xs, 5),
        "RBF": rbf_features(xs, centers, sigma),
        "tile coding (3 tilings)": tile_features(xs, centers, width, offsets),
        "coarse coding (3 fields)": coarse_features(xs, centers, width, offsets),
        "state aggregation": aggregation_features(xs, centers),
    }


def run(name: str, y: np.ndarray, rows: list[dict], n_centers: int, plot: bool) -> None:
    feats = build_features(n_centers)
    fig, axs = plt.subplots(2, len(feats), figsize=(4.0 * len(feats), 7.2))
    for c, (label, phi) in enumerate(feats.items()):
        pred, history, iters = fit(phi, y, alpha=0.5)
        mse = float(np.mean((y - pred) ** 2))
        rows.append(dict(target=name, n_centers=n_centers, features=label,
                         n_features=phi.shape[1], iterations=iters, mse=mse))
        print(f"    {label:<26} {phi.shape[1]:>4} features  "
              f"{iters:>6} iters  MSE {mse:12.4f}")

        axs[0][c].plot(X, y, lw=2, label="target", color="0.3")
        axs[0][c].plot(X, pred, lw=1.5, label="fit", color="#d62728")
        axs[0][c].set_title(f"{label}\nMSE {mse:.3g}", fontsize=9)
        axs[0][c].legend(fontsize=8)
        axs[0][c].grid(alpha=0.3)
        axs[1][c].plot(history, color="#1f77b4")
        axs[1][c].set_yscale("log")
        axs[1][c].set_xlabel("gradient step")
        axs[1][c].grid(alpha=0.3)
        if c == 0:
            axs[0][c].set_ylabel("value")
            axs[1][c].set_ylabel("MSE")
    fig.suptitle(f"Fitting the {name} target, {n_centers} centres for "
                 f"{len(X)} samples", fontsize=13)
    if plot:
        save_fig(fig, HERE / "figures" / f"regression_{name}.png")
    else:
        plt.close(fig)


def main() -> None:
    rows: list[dict] = []
    for n_centers in (20, 100):
        for name, y in (("smooth", target_smooth()), ("piecewise", target_piecewise())):
            print(f"  target: {name}, {n_centers} centres for {len(X)} samples")
            run(name, y, rows, n_centers, plot=(n_centers == 20))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / "regression_mse.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out/'regression_mse.csv'}")


if __name__ == "__main__":
    main()
