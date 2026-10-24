"""Policy iteration, value iteration and generalized policy iteration.

Task
----
Implement PI, VI and GPI as in the book, learning both v_* (part 1) and q_*
(part 2), from seven initial values {-100, -10, -5, 0, 5, 10, 100}, with
gamma = 0.99.  Log the total absolute Bellman error at every policy
evaluation step, count how many policy evaluation steps each algorithm needs,
report mean and standard deviation across the seven initialisations, and
check every learned policy against the hand written optimal policy.

Reference
---------
Sutton and Barto, 2nd edition.  Policy iteration is the boxed algorithm in
section 4.3, value iteration is section 4.4 and equation 4.10, generalized
policy iteration is section 4.6.

Counting, and why it is the whole point
---------------------------------------
"Number of policy evaluation steps" only means something if all three
algorithms count the same event.  Here one step is **one application of a
Bellman backup to the whole state space**, and exactly one Bellman error is
logged per step, for all three algorithms and for both V and Q.  Under that
definition:

* PI runs evaluation to convergence, improves once, repeats.  Its count is
  the sum of the evaluation sweeps over all its improvement rounds.
* VI applies the optimality backup once per step.  Each sweep is one step.
* GPI runs five evaluation sweeps, improves once, repeats.

Logging the error inside the per state loop instead would produce one entry
per state per sweep, inflating the count by a factor of |S| and making the
three algorithms incomparable.  All three also use a synchronous sweep:
in-place Gauss-Seidel updates converge faster, so mixing the two styles would
turn the comparison partly into a comparison of update style.  See
docs/ENGINEERING_NOTES.md section 5.

Outputs
-------
figures/bellman_error_V.png   3 algorithms x 7 initialisations, learning v_*
figures/bellman_error_Q.png   3 algorithms x 7 initialisations, learning q_*
results/iteration_counts.csv  every count
results/summary_table.md      the mean +/- std table
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit.bellman import (  # noqa: E402
    bellman_error,
    expected_q_backup,
    expected_v_backup,
    greedy_policy_from_q,
    greedy_policy_from_v,
    optimal_q_backup,
    optimal_v_backup,
)
from rlkit.envs import MDP, extract_mdp, policy_from_names  # noqa: E402
from rlkit.plotting import save_fig  # noqa: E402

HERE = Path(__file__).resolve().parent
GAMMA = 0.99
THETA = 1e-10
MAX_STEPS = 100_000
INIT_VALUES = [-100.0, -10.0, -5.0, 0.0, 5.0, 10.0, 100.0]

OPTIMAL_POLICY_NAMES = [
    ["DOWN", "RIGHT", "STAY"],
    ["DOWN", "RIGHT", "UP"],
    ["RIGHT", "RIGHT", "UP"],
]


# ---------------------------------------------------------------------------
# the three algorithms, written once each for V and once each for Q
#
# `backup` applies one expectation sweep, `optimal_backup` one optimality
# sweep, `greedy` extracts the greedy policy.  Passing those three in is what
# lets the V and Q versions share the control flow exactly, so that a
# difference in the results is a difference in the value representation and
# not a difference in the code.
# ---------------------------------------------------------------------------

def _uniform_policy(mdp: MDP) -> np.ndarray:
    """The random policy the assignment asks PI to start from."""
    return np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)


def policy_iteration(mdp, gamma, init_value, backup, greedy, init_shape):
    """Evaluate to convergence, improve, repeat until the policy stops changing."""
    value = np.full(init_shape, float(init_value))
    pi = _uniform_policy(mdp)
    errors: list[float] = []

    while len(errors) < MAX_STEPS:
        # policy evaluation, run to convergence
        while len(errors) < MAX_STEPS:
            new_value = backup(value, pi, mdp, gamma)
            err = bellman_error(value, new_value)
            errors.append(err)
            value = new_value
            if err < THETA:
                break
        # policy improvement
        new_pi = greedy(value, mdp, gamma)
        if np.array_equal(new_pi, pi):
            break
        pi = new_pi
    return pi, value, errors


def value_iteration(mdp, gamma, init_value, optimal_backup, greedy, init_shape):
    """One optimality sweep per step until the Bellman error falls below theta."""
    value = np.full(init_shape, float(init_value))
    errors: list[float] = []
    while len(errors) < MAX_STEPS:
        new_value = optimal_backup(value, mdp, gamma)
        err = bellman_error(value, new_value)
        errors.append(err)
        value = new_value
        if err < THETA:
            break
    return greedy(value, mdp, gamma), value, errors


def generalized_policy_iteration(
    mdp, gamma, init_value, backup, greedy, init_shape, eval_steps: int = 5
):
    """Five evaluation sweeps, one improvement, repeat.

    Stopping rule: the policy is unchanged by an improvement *and* the last
    evaluation sweep was below theta.  Requiring both is what makes the count
    comparable with PI and VI.  Stopping on policy stability alone would let
    GPI finish with a value function that has not converged, which would make
    its iteration count look good for a reason that has nothing to do with
    the algorithm.
    """
    value = np.full(init_shape, float(init_value))
    pi = _uniform_policy(mdp)
    errors: list[float] = []

    while len(errors) < MAX_STEPS:
        err = np.inf
        for _ in range(eval_steps):
            new_value = backup(value, pi, mdp, gamma)
            err = bellman_error(value, new_value)
            errors.append(err)
            value = new_value
            if err < THETA:
                break
        new_pi = greedy(value, mdp, gamma)
        stable = np.array_equal(new_pi, pi)
        pi = new_pi
        if stable and err < THETA:
            break
    return pi, value, errors


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def _v_adapters(mdp):
    return dict(
        backup=expected_v_backup,
        optimal_backup=optimal_v_backup,
        greedy=greedy_policy_from_v,
        init_shape=(mdp.n_states,),
    )


def _q_adapters(mdp):
    return dict(
        backup=expected_q_backup,
        optimal_backup=optimal_q_backup,
        # greedy_policy_from_q ignores mdp and gamma, wrap it to the same signature
        greedy=lambda Q, _mdp, _gamma: greedy_policy_from_q(Q),
        init_shape=(mdp.n_states, mdp.n_actions),
    )


def run(mdp, which: str, pi_opt: np.ndarray):
    """Run all three algorithms from all seven initialisations."""
    ad = _v_adapters(mdp) if which == "V" else _q_adapters(mdp)
    curves: dict[str, dict[float, list[float]]] = {}
    counts: dict[str, dict[float, int]] = {}

    specs = {
        "VI": lambda iv: value_iteration(
            mdp, GAMMA, iv, ad["optimal_backup"], ad["greedy"], ad["init_shape"]),
        "PI": lambda iv: policy_iteration(
            mdp, GAMMA, iv, ad["backup"], ad["greedy"], ad["init_shape"]),
        "GPI": lambda iv: generalized_policy_iteration(
            mdp, GAMMA, iv, ad["backup"], ad["greedy"], ad["init_shape"]),
    }

    for name, fn in specs.items():
        curves[name], counts[name] = {}, {}
        for iv in INIT_VALUES:
            pi, _value, errors = fn(iv)
            assert np.allclose(pi, pi_opt), (
                f"{which} {name} with init {iv} did not recover the optimal policy"
            )
            curves[name][iv] = errors
            counts[name][iv] = len(errors)
    return curves, counts


def plot(curves, which, out_path):
    algs = ["VI", "PI", "GPI"]
    # deliberately not sharing axes: policy iteration needs thousands of
    # evaluation steps while value iteration needs eight, so a shared x-axis
    # compresses the VI and GPI curves into an invisible vertical line.  The
    # step count is printed in each panel instead.
    fig, axs = plt.subplots(3, len(INIT_VALUES), figsize=(2.6 * len(INIT_VALUES), 7.6))
    fig.suptitle(
        f"Total absolute Bellman error while learning ${'v' if which == 'V' else 'q'}_*$, "
        f"$\\gamma = {GAMMA}$",
        fontsize=13,
    )
    for r, alg in enumerate(algs):
        for c, iv in enumerate(INIT_VALUES):
            e = np.asarray(curves[alg][iv])
            ax = axs[r][c]
            # a Bellman error of exactly 0 cannot be drawn on a log axis; clip
            # it to the smallest positive value so the final point stays visible
            pos = e[e > 0]
            floor = pos.min() * 0.5 if pos.size else 1e-16
            ax.plot(np.arange(1, len(e) + 1), np.maximum(e, floor), lw=1.2)
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(f"$init = {iv:g}$", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{alg}\nBellman error", fontsize=9)
            if r == 2:
                ax.set_xlabel("evaluation step", fontsize=8)
            ax.text(0.95, 0.9, f"n={len(e)}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7)
    save_fig(fig, out_path)


def theta_sensitivity(mdp, pi_opt) -> list[str]:
    """How the comparison moves when the convergence tolerance moves.

    This is reported because the ranking of the three algorithms is not a
    property of the algorithms alone.  VI and GPI reach an exactly zero
    Bellman error here and so are insensitive to theta, while PI's evaluation
    of the uniform random policy converges only geometrically and its cost
    grows without bound as theta shrinks.
    """
    global THETA
    original = THETA
    lines = ["\n### Sensitivity of the comparison to the tolerance\n",
             "Mean policy evaluation steps over the seven initialisations, learning `v_*`.\n",
             "| theta | VI | PI | GPI |", "|---|---|---|---|"]
    for th in (1e-4, 1e-6, 1e-8, 1e-10):
        THETA = th
        _, counts = run(mdp, "V", pi_opt)
        means = {a: np.mean([counts[a][iv] for iv in INIT_VALUES]) for a in ("VI", "PI", "GPI")}
        lines.append(f"| {th:g} | {means['VI']:.1f} | {means['PI']:.1f} | {means['GPI']:.1f} |")
        print(f"theta={th:<7g} VI {means['VI']:7.1f}  PI {means['PI']:7.1f}  GPI {means['GPI']:7.1f}")
    THETA = original
    return lines


def main() -> None:
    mdp = extract_mdp("Gym-Gridworlds/Penalty-3x3-v0")
    idx = policy_from_names(OPTIMAL_POLICY_NAMES)
    pi_opt = np.zeros((mdp.n_states, mdp.n_actions))
    pi_opt[np.arange(mdp.n_states), idx] = 1.0

    rows, table_lines = [], []
    for which in ("V", "Q"):
        curves, counts = run(mdp, which, pi_opt)
        plot(curves, which, HERE / "figures" / f"bellman_error_{which}.png")

        table_lines.append(f"\n### Learning `{'v' if which == 'V' else 'q'}_*`\n")
        header = "| algorithm | " + " | ".join(f"{iv:g}" for iv in INIT_VALUES) + " | mean +/- std |"
        table_lines += [header, "|" + "---|" * (len(INIT_VALUES) + 2)]
        for alg in ("VI", "PI", "GPI"):
            n = np.array([counts[alg][iv] for iv in INIT_VALUES], dtype=float)
            table_lines.append(
                f"| {alg} | " + " | ".join(f"{int(x)}" for x in n)
                + f" | **{n.mean():.1f} +/- {n.std():.1f}** |"
            )
            for iv in INIT_VALUES:
                rows.append(dict(value=which, algorithm=alg, init=iv,
                                 evaluation_steps=counts[alg][iv]))
            print(f"{which} {alg:<3}  {n.astype(int).tolist()}  "
                  f"mean {n.mean():.1f} +/- {n.std():.1f}")

    table_lines += theta_sensitivity(mdp, pi_opt)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / "iteration_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary_table.md", "w") as f:
        f.write("# Policy evaluation steps to convergence\n")
        f.write(f"\nOne step = one Bellman backup over the whole state space. "
                f"gamma = {GAMMA}, theta = {THETA:g}.\n")
        f.write("\n".join(table_lines) + "\n")
    print(f"  wrote {out/'summary_table.md'}")


if __name__ == "__main__":
    main()
