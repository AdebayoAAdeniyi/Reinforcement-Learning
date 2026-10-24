"""Parts 3 and 4: batch semi-gradient TD prediction of v_pi and q_pi.

Reference
---------
Sutton and Barto, 2nd edition, section 9.3.  The update is equation 9.7 with
the TD(0) target,

    w <- w + alpha [ R + gamma v(S', w) - v(S, w) ] grad v(S, w)

run in batch over the whole dataset rather than one sample at a time.

What "semi-gradient" means
--------------------------
The objective is the mean squared error between v(S, w) and the target
`R + gamma v(S', w)`.  The target depends on `w` too, so the true gradient of
that objective would include a `- gamma grad v(S', w)` term.  Semi-gradient
methods simply leave that term out and treat the target as a constant.  That
is the whole content of the word: the update follows only part of the
gradient.

It has nothing to do with using a subset of the data.  Using a subset is
stochastic gradient descent, a separate and independent choice that can be
combined with either a full gradient or a semi-gradient.  The two are
routinely confused, and the distinction has teeth: a semi-gradient update is
not the gradient of any objective, so the usual convergence guarantees of
gradient descent do not apply.  On-policy linear TD, which is what runs here,
still converges, but to the TD fixed point rather than to the minimum mean
squared error (section 9.4); off-policy linear TD can diverge outright
(section 11.2).

Part 4 changes exactly one thing
--------------------------------
The weights become a (D, A) matrix instead of a (D,) vector, the prediction
for a transition is read from the column of the action taken, the target
bootstraps from `sum_a' pi(a'|S') q(S', a', w)`, and the gradient is written
back only into that one column.  Nothing else about the algorithm changes.

Outputs
-------
figures/td_v_comparison.png   all five feature constructions against true v_pi
figures/td_q_heatmaps.png     true and predicted q_pi, one row per action
results/td_prediction_mse.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit.envs import ACTION_NAMES  # noqa: E402
from rlkit.features import (  # noqa: E402
    aggregation_features, coarse_features, make_centers, poly_features,
    rbf_features, tile_features,
)
from rlkit.plotting import save_fig  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dangermaze.npz"
MAX_ITER = 20_000
THRESH = 1e-10
ALPHA = 0.3


def load():
    if not DATA.exists():
        raise SystemExit("dataset missing, run `python make_dataset.py` first")
    d = np.load(DATA)
    return d, tuple(int(v) for v in d["grid"])


def build_feature_functions(grid):
    """The five constructions over the 2D grid coordinates.

    Centres sit on a lattice coarser than the grid itself, 4 by 4 against the
    5 by 6 of the environment, so that none of these can simply memorise one
    value per state.  With one centre per state, aggregation becomes a lookup
    table and the comparison is vacuous.
    """
    low, high = [0.0, 0.0], [grid[0] - 1.0, grid[1] - 1.0]
    centers = make_centers(low, high, [4, 4], pad=0.1)
    span = np.array([grid[0] - 1.0, grid[1] - 1.0])
    width = span / 4.0 * 2.0
    sigma = span / 4.0 * 0.9
    offsets = [(-width[0] / 4, 0.0), (0.0, width[1] / 4), (width[0] / 4, 0.0)]
    return {
        "polynomial (deg 3)": lambda s: poly_features(s, 3),
        "RBF": lambda s: rbf_features(s, centers, sigma),
        "tile coding (3 tilings)": lambda s: tile_features(s, centers, width, offsets),
        "coarse coding (3 fields)": lambda s: coarse_features(s, centers, width, offsets),
        "state aggregation": lambda s: aggregation_features(s, centers),
    }


def batch_semi_gradient_td_v(phi, phi_next, r, term, gamma, alpha=ALPHA):
    """Batch semi-gradient TD(0) for the state value function."""
    n, d = phi.shape
    scale = max(np.linalg.eigvalsh(phi.T @ phi / n).max(), 1e-12)
    step = alpha / scale
    w = np.zeros(d)
    history = []
    for it in range(MAX_ITER):
        v = phi @ w
        # the target is treated as a constant: this is the semi-gradient
        target = r + (1.0 - term) * gamma * (phi_next @ w)
        delta = target - v
        history.append(float(np.mean(delta ** 2)))
        w = w + step * (phi.T @ delta) / n
        if it > 0 and abs(history[-2] - history[-1]) < THRESH:
            break
    return w, np.array(history)


def batch_semi_gradient_td_q(phi, phi_next, a, r, term, pi_next, gamma,
                             n_actions, alpha=ALPHA):
    """Batch semi-gradient TD(0) for the action value function.

    `pi_next` has shape (N, A) and holds pi(. | S') for every transition, so
    the bootstrap is the expectation under the policy being evaluated.  This
    is TD prediction of q_pi, the action value analogue of the V version, not
    Q-learning: taking a max here would estimate q_* instead and would not be
    comparable with part 3.
    """
    n, d = phi.shape
    scale = max(np.linalg.eigvalsh(phi.T @ phi / n).max(), 1e-12)
    step = alpha / scale
    W = np.zeros((d, n_actions))
    rows = np.arange(n)
    history = []
    for it in range(MAX_ITER):
        q_taken = (phi @ W)[rows, a]
        target = r + (1.0 - term) * gamma * ((phi_next @ W) * pi_next).sum(axis=1)
        delta = target - q_taken
        history.append(float(np.mean(delta ** 2)))
        # scatter the gradient into the column of the action actually taken
        grad = np.zeros_like(W)
        for act in range(n_actions):
            m = a == act
            if m.any():
                grad[:, act] = phi[m].T @ delta[m] / n
        W = W + step * grad
        if it > 0 and abs(history[-2] - history[-1]) < THRESH:
            break
    return W, np.array(history)


def heat(ax, values, grid, title, vmin, vmax):
    im = ax.imshow(values.reshape(grid), cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in range(values.size):
        ax.text(s % grid[1], s // grid[1], f"{values[s]:.0f}",
                ha="center", va="center", fontsize=6.5, color="k")
    return im


def main() -> None:
    d, grid = load()
    gamma = float(d["gamma"])
    s, s_next = d["s"], d["s_next"]
    a, r, term = d["a"].astype(int), d["r"], d["term"]
    pi, V_pi, Q_pi = d["pi"], d["V_pi"], d["Q_pi"]
    n_states, n_actions = Q_pi.shape

    all_states = np.stack(np.unravel_index(np.arange(n_states), grid), axis=1).astype(float)
    pi_next = pi[d["s_next_idx"].astype(int)]

    print(f"  {len(s)} transitions, gamma {gamma}, "
          f"behaviour epsilon {float(d['behaviour_eps'])}")

    rows, preds = [], {}
    for name, get_phi in build_feature_functions(grid).items():
        w, hist = batch_semi_gradient_td_v(get_phi(s), get_phi(s_next), r, term, gamma)
        V_hat = get_phi(all_states) @ w
        mse = float(np.mean((V_hat - V_pi) ** 2))
        preds[name] = V_hat
        rows.append(dict(part="V", features=name, n_features=w.size,
                         iterations=len(hist), mse_vs_true=mse))
        print(f"    V  {name:<26} {w.size:>3} features  MSE vs true v_pi {mse:10.3f}")

    best = min(preds, key=lambda k: np.mean((preds[k] - V_pi) ** 2))
    print(f"  best for V: {best}")

    vmin, vmax = V_pi.min(), V_pi.max()
    fig, axs = plt.subplots(1, len(preds) + 1, figsize=(3.1 * (len(preds) + 1), 3.6))
    heat(axs[0], V_pi, grid, "true $v_\\pi$", vmin, vmax)
    for ax, (name, V_hat) in zip(axs[1:], preds.items()):
        heat(ax, V_hat, grid, f"{name}\nMSE {np.mean((V_hat - V_pi) ** 2):.1f}", vmin, vmax)
    fig.suptitle("Batch semi-gradient TD prediction of $v_\\pi$ on DangerMaze-6x6",
                 fontsize=13)
    save_fig(fig, HERE / "figures" / "td_v_comparison.png")

    # part 4: the same algorithm for q_pi, with the construction that won
    get_phi = build_feature_functions(grid)[best]
    W, hist = batch_semi_gradient_td_q(get_phi(s), get_phi(s_next), a, r, term,
                                       pi_next, gamma, n_actions)
    Q_hat = get_phi(all_states) @ W
    mse_q = float(np.mean((Q_hat - Q_pi) ** 2))
    rows.append(dict(part="Q", features=best, n_features=W.size,
                     iterations=len(hist), mse_vs_true=mse_q))
    print(f"    Q  {best:<26} {W.size:>3} weights   MSE vs true q_pi {mse_q:10.3f}")

    vmin, vmax = Q_pi.min(), Q_pi.max()
    fig, axs = plt.subplots(2, n_actions, figsize=(2.6 * n_actions, 6.0))
    for act in range(n_actions):
        heat(axs[0][act], Q_pi[:, act], grid, f"true, {ACTION_NAMES[act]}", vmin, vmax)
        heat(axs[1][act], Q_hat[:, act], grid, f"predicted, {ACTION_NAMES[act]}", vmin, vmax)
    fig.suptitle(f"Batch semi-gradient TD prediction of $q_\\pi$, {best}, "
                 f"MSE {mse_q:.1f}", fontsize=13)
    save_fig(fig, HERE / "figures" / "td_q_heatmaps.png")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / "td_prediction_mse.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    print(f"  wrote {out/'td_prediction_mse.csv'}")


if __name__ == "__main__":
    main()
