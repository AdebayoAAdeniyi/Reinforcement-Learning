"""Plotting helpers shared by every chapter.

Matplotlib is forced onto the Agg backend here.  `plt.ion()` and `plt.show()`
block or silently draw nothing on a machine without a display, which includes
continuous integration and most remote shells.  Every script writes its
figures to disk instead, so `python <script>.py` behaves the same
everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["smooth", "error_shade_plot", "save_fig", "figure_dir", "CYCLE"]

#: A colour cycle long enough for the nine curve sweeps in chapters 04 and 06.
CYCLE = [
    "#d62728", "#2ca02c", "#1f77b4", "#000000", "#ff7f0e",
    "#17becf", "#8c564b", "#7f7f7f", "#e377c2",
]


def smooth(arr: np.ndarray, span: int) -> np.ndarray:
    """Centred moving average, with the ends handled by a shrinking window.

    Adapted from https://stackoverflow.com/a/63458548/754136
    """
    arr = np.asarray(arr, dtype=float)
    out = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    out[0] = arr[0]
    for i in range(1, span + 1):
        out[i] = np.average(arr[: i + span])
        out[-i] = np.average(arr[-i - span:])
    return out


def error_shade_plot(ax, data, stepsize=1, smoothing_window=1, **kwargs):
    """Mean curve across seeds with a 95 percent confidence band.

    `data` has shape (n_seeds, n_points).  The band is 1.96 standard errors,
    so it is a confidence interval on the mean across seeds, not the spread of
    the seeds themselves.  That distinction matters when reading the chapter
    04 and 07 figures: a narrow band means the mean is well determined, not
    that individual runs agree.
    """
    data = np.asarray(data, dtype=float)
    y = np.nanmean(data, axis=0)
    x = np.arange(len(y)) * stepsize
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    err = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        err = smooth(err, smoothing_window)
    err = 1.96 * err / np.sqrt(data.shape[0])
    ax.fill_between(x, y - err, y + err, alpha=0.2, linewidth=0.0, color=line.get_color())
    return line


def figure_dir(script_file: str) -> Path:
    """`figures/` next to the script that is running, created if missing."""
    d = Path(script_file).resolve().parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, path, dpi: int = 130) -> Path:
    """Save and close, and print the path so a run leaves a readable trail."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")
    return path
