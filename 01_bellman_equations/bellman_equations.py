"""Bellman expectation equations on Penalty-3x3: v_pi and q_pi by policy evaluation.

Task
----
Compute v_pi and q_pi of a manually specified optimal policy on
`Gym-Gridworlds/Penalty-3x3-v0` by iterative policy evaluation, logging the
total absolute Bellman error after every value update, for every combination
of gamma in {0.01, 0.5, 0.99} and initial value in {-10, 0, 10}.

Reference
---------
Sutton and Barto, 2nd edition, equations 3.12 to 3.14 and section 4.1.

    v_pi(s)   = sum_a pi(a|s) sum_{s',r} p(s',r|s,a) [ r + gamma v_pi(s') ]
    q_pi(s,a) =               sum_{s',r} p(s',r|s,a) [ r + gamma sum_a' pi(a'|s') q_pi(s',a') ]

Both operators live in `rlkit.bellman` so that the dynamic programming chapter
uses literally the same code.

Outputs
-------
figures/v_pi_init_{-10,0,10}.png   heatmap of v_pi and the Bellman error trend
figures/q_pi_init_{-10,0,10}.png   one heatmap per action plus the error trend
results/bellman_errors.csv         every logged Bellman error, long format
results/value_functions.csv        the converged v_pi and q_pi
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit.bellman import policy_evaluation_q, policy_evaluation_v  # noqa: E402
from rlkit.envs import ACTION_NAMES, extract_mdp, policy_from_names  # noqa: E402
from rlkit.plotting import save_fig  # noqa: E402

HERE = Path(__file__).resolve().parent
GRID = (3, 3)
GAMMAS = [0.01, 0.5, 0.99]
INIT_VALUES = [-10.0, 0.0, 10.0]

# The optimal policy, written out by hand as the assignment asks.  It is given
# by action *name* rather than by index on purpose: the integer encoding of
# these five actions changed in `gym_gridworlds` after this course ran, so the
# equivalent integer array means something different depending on which commit
# is installed.  See rlkit/envs.py and docs/ENGINEERING_NOTES.md section 1.
#
#     0: DOWN   1: RIGHT  2: STAY     the agent walks down the left column,
#     3: DOWN   4: RIGHT  5: UP       along the bottom, up the right column
#     6: RIGHT  7: RIGHT  8: UP       and stays on the green tile
#
# The route 0 -> 3 -> 6 -> 7 -> 8 -> 5 -> 2 avoids both red tiles (states 1
# and 4) and takes six moves, so v_pi(0) = 0.99^6 = 0.941 when gamma = 0.99.
OPTIMAL_POLICY_NAMES = [
    ["DOWN", "RIGHT", "STAY"],
    ["DOWN", "RIGHT", "UP"],
    ["RIGHT", "RIGHT", "UP"],
]


def deterministic_policy_matrix(action_indices: np.ndarray, n_actions: int) -> np.ndarray:
    """Turn a vector of action indices into a one hot (S, A) policy matrix."""
    pi = np.zeros((len(action_indices), n_actions))
    pi[np.arange(len(action_indices)), action_indices] = 1.0
    return pi


def plot_v(results, init_value, out_path):
    """Heatmap of v_pi and the Bellman error trend, one column per gamma."""
    fig, axs = plt.subplots(2, len(GAMMAS), figsize=(4.0 * len(GAMMAS), 6.4))
    fig.suptitle(f"$v_\\pi$ of the optimal policy, $V_0 = {init_value:g}$", fontsize=13)
    for i, gamma in enumerate(GAMMAS):
        V, errors = results[gamma]
        im = axs[0][i].imshow(V.reshape(GRID), cmap="viridis")
        axs[0][i].set_title(f"$\\gamma$ = {gamma}")
        axs[0][i].set_xticks(range(GRID[1]))
        axs[0][i].set_yticks(range(GRID[0]))
        for s in range(len(V)):
            axs[0][i].text(
                s % GRID[1], s // GRID[1], f"{V[s]:.2f}",
                ha="center", va="center", color="w", fontsize=9,
            )
        fig.colorbar(im, ax=axs[0][i], fraction=0.046)

        # the last error is often exactly 0, which a log axis cannot draw, so
        # it is clipped to half the smallest positive value and the point
        # stays visible.  The panel title gives the true iteration count.
        e = np.asarray(errors)
        pos = e[e > 0]
        floor = pos.min() * 0.5 if pos.size else 1e-16
        axs[1][i].plot(np.arange(1, len(e) + 1), np.maximum(e, floor),
                       marker="o", ms=3)
        axs[1][i].set_yscale("log")
        axs[1][i].set_xlabel("policy evaluation iteration")
        axs[1][i].grid(alpha=0.3)
        if i == 0:
            axs[1][i].set_ylabel("total absolute Bellman error")
        axs[1][i].set_title(f"converged in {len(errors)} iterations")
    save_fig(fig, out_path)


def plot_q(results, init_value, out_path):
    """One heatmap per action plus the Bellman error trend, per gamma."""
    n_actions = len(ACTION_NAMES)
    fig, axs = plt.subplots(n_actions + 1, len(GAMMAS),
                            figsize=(3.4 * len(GAMMAS), 2.3 * (n_actions + 1)))
    fig.suptitle(f"$q_\\pi$ of the optimal policy, $Q_0 = {init_value:g}$", fontsize=13)
    for i, gamma in enumerate(GAMMAS):
        Q, errors = results[gamma]
        vmin, vmax = Q.min(), Q.max()
        for a in range(n_actions):
            ax = axs[a][i]
            ax.imshow(Q[:, a].reshape(GRID), cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(ACTION_NAMES[a], fontsize=9)
            if a == 0:
                ax.set_title(f"$\\gamma$ = {gamma}")
            for s in range(Q.shape[0]):
                ax.text(s % GRID[1], s // GRID[1], f"{Q[s, a]:.2f}",
                        ha="center", va="center", color="w", fontsize=7)
        e = np.asarray(errors)
        pos = e[e > 0]
        floor = pos.min() * 0.5 if pos.size else 1e-16
        axs[-1][i].plot(np.arange(1, len(e) + 1), np.maximum(e, floor),
                        marker="o", ms=3)
        axs[-1][i].set_yscale("log")
        axs[-1][i].set_xlabel("policy evaluation iteration")
        axs[-1][i].grid(alpha=0.3)
        if i == 0:
            axs[-1][i].set_ylabel("Bellman error", fontsize=9)
    save_fig(fig, out_path)


def main() -> None:
    mdp = extract_mdp("Gym-Gridworlds/Penalty-3x3-v0")
    pi_indices = policy_from_names(OPTIMAL_POLICY_NAMES)
    pi = deterministic_policy_matrix(pi_indices, mdp.n_actions)

    print("optimal policy (by name):")
    for row in OPTIMAL_POLICY_NAMES:
        print("   " + "  ".join(f"{a:>5s}" for a in row))
    print(f"   as indices under the installed package: {pi_indices.tolist()}")

    error_rows, value_rows = [], []

    for init_value in INIT_VALUES:
        v_results, q_results = {}, {}
        for gamma in GAMMAS:
            V, v_err = policy_evaluation_v(pi, mdp, gamma, init_value=init_value)
            Q, q_err = policy_evaluation_q(pi, mdp, gamma, init_value=init_value)
            v_results[gamma], q_results[gamma] = (V, v_err), (Q, q_err)

            # v_pi(s) must equal sum_a pi(a|s) q_pi(s,a).  Asserting it
            # here rather than trusting it catches an inconsistent pair of
            # operators at the point of use; the same identity is tested
            # against random policies in tests/test_rlkit.py.
            assert np.allclose(V, (pi * Q).sum(axis=1), atol=1e-8), (
                "v_pi and q_pi disagree, the two Bellman operators are inconsistent"
            )

            for it, e in enumerate(v_err, start=1):
                error_rows.append(dict(value="v", gamma=gamma, init=init_value,
                                       iteration=it, bellman_error=e))
            for it, e in enumerate(q_err, start=1):
                error_rows.append(dict(value="q", gamma=gamma, init=init_value,
                                       iteration=it, bellman_error=e))
            for s in range(mdp.n_states):
                row = dict(gamma=gamma, init=init_value, state=s, v=V[s])
                row.update({f"q_{ACTION_NAMES[a]}": Q[s, a] for a in range(mdp.n_actions)})
                value_rows.append(row)

            print(f"  gamma={gamma:<5} V0={init_value:>5.0f}  "
                  f"v sweeps {len(v_err):>3d}  q sweeps {len(q_err):>3d}  "
                  f"v_pi(s0)={V[0]: .4f}")

        plot_v(v_results, init_value, HERE / "figures" / f"v_pi_init_{init_value:g}.png")
        plot_q(q_results, init_value, HERE / "figures" / f"q_pi_init_{init_value:g}.png")

    results = HERE / "results"
    results.mkdir(exist_ok=True)
    with open(results / "bellman_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(error_rows[0]))
        w.writeheader()
        w.writerows(error_rows)
    with open(results / "value_functions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(value_rows[0]))
        w.writeheader()
        w.writerows(value_rows)
    print(f"  wrote {len(error_rows)} Bellman error records and "
          f"{len(value_rows)} value records to results/")


if __name__ == "__main__":
    main()
