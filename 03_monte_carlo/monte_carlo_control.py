"""Monte Carlo control, on-policy and off-policy with weighted importance sampling.

Task
----
Part 1: on-policy every visit MC control on `Penalty-3x3-v0`, sweeping the
number of episodes collected per update (1, 10, 50) and the speed at which
epsilon decays (1, 2, 5).  Part 2: the same with reward noise of std 3.0.
Part 3: off-policy MC control with weighted importance sampling, where the
target policy is greedier than the behaviour policy.

Reference
---------
Sutton and Barto, 2nd edition, chapter 5.  Every visit MC is section 5.1,
the on-policy epsilon-soft control algorithm is the box in section 5.4,
weighted importance sampling is equation 5.6 and the incremental form is
equation 5.8, and the off-policy MC control algorithm is the box in
section 5.7.

The weighted importance sampling update
---------------------------------------
Walking a trajectory backwards from the last step, with behaviour policy b
and target policy pi,

    G      <- r_t + gamma G
    C(s,a) <- C(s,a) + W
    Q(s,a) <- Q(s,a) + (W / C(s,a)) [ G - Q(s,a) ]
    W      <- W * pi(a|s) / b(a|s)
    if W == 0: break

Three details decide whether this is weighted importance sampling or an
expensive way of computing a plain average, and all three are easy to get
subtly wrong:

1.  `C` is a *cumulative* sum of weights that persists across episodes, so it
    is created once outside the episode loop.  Rebuilt per episode it resets
    to zero, the step size `W / C` becomes 1, and all history is discarded.
2.  `W` accumulates along the trajectory as a product of per step ratios and
    is updated *after* it is used.  A single step ratio recomputed from
    scratch is not an importance sampling ratio.
3.  The behaviour probability `b(a|s)` is recorded at the moment the action
    was taken, not recomputed afterwards from an already updated `Q`, which
    would give the ratio of two different behaviour policies.

Outputs
-------
figures/mc_on_policy.png      part 1 and 2, no noise and noise std 3.0
figures/mc_importance.png     part 3, the same sweep off-policy
figures/mc_on_vs_off.png      the two put side by side at a fixed setting
results/final_bellman_error.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit.bellman import (  # noqa: E402
    bellman_error, greedy_policy_from_q, solve_q_exactly, solve_v_exactly,
)
from rlkit.envs import extract_mdp, make_env, reset_seeded  # noqa: E402
from rlkit.plotting import CYCLE, error_shade_plot, save_fig  # noqa: E402
from rlkit.policies import eps_greedy_probs, eps_greedy_probs_all  # noqa: E402

HERE = Path(__file__).resolve().parent
ENV_ID = "Gym-Gridworlds/Penalty-3x3-v0"
GAMMA = 0.9
HORIZON = 10
MAX_STEPS = 2000
EPISODES_PER_ITERATION = [1, 10, 50]
DECAYS = [1, 2, 5]
N_SEEDS = 50
EPS_MIN = 0.01
TARGET_EPS = 0.1      # the target policy is greedier than the behaviour policy


def collect_episodes(env, Q, eps, n_episodes, rng, episode_seed):
    """Run `n_episodes` under the epsilon-greedy behaviour policy."""
    data = {"s": [], "a": [], "r": [], "b": []}
    for i in range(n_episodes):
        s, _ = reset_seeded(env, episode_seed + i)
        done = False
        while not done:
            probs = eps_greedy_probs(Q[s], eps)
            a = int(rng.choice(Q.shape[1], p=probs))
            s_next, r, terminated, truncated, _ = env.step(a)
            data["s"].append(s)
            data["a"].append(a)
            data["r"].append(r)
            # the probability under which this action was actually taken.
            # Recomputing it later from an already updated Q would give the
            # ratio of two different behaviour policies, not an importance
            # sampling ratio.
            data["b"].append(probs[a])
            s, done = s_next, terminated or truncated
    return data


def monte_carlo(env, mdp, Q, gamma, decay, max_steps, episodes_per_iteration,
                off_policy, seed):
    """Every visit MC control, on-policy or off-policy with weighted IS.

    Returns the final Q and one Bellman error per environment step, so that
    curves from different `episodes_per_iteration` settings are comparable:
    they all consume the same budget of environment interaction.
    """
    rng = np.random.default_rng(seed)
    n_states, n_actions = Q.shape
    eps = 1.0

    # running every visit average, used on-policy
    counts = np.zeros_like(Q)
    # cumulative importance weights, used off-policy.  Created once, outside
    # the episode loop, because it has to accumulate across episodes.
    C = np.zeros_like(Q)

    be_per_step = np.full(max_steps, np.nan)
    # the Bellman error above is measured against q_pi of a policy that is
    # itself changing as epsilon decays, so it is a moving reference.  As
    # epsilon shrinks, q_pi of the near greedy policy becomes extreme in
    # states the policy would loiter in, and the error can rise while the
    # policy is still improving.  This second metric is computed against a
    # fixed reference instead: the exact value, under the true model, of the
    # start state under the policy that is greedy with respect to the current
    # Q.  Its maximum is v_*(s0) = 0.9^6 = 0.531 at gamma = 0.9.
    greedy_v0_per_step = np.full(max_steps, np.nan)
    tot_steps = 0
    episode_seed = seed * 100_000

    while tot_steps < max_steps:
        data = collect_episodes(env, Q, eps, episodes_per_iteration, rng, episode_seed)
        episode_seed += episodes_per_iteration
        n_new = len(data["s"])

        if off_policy:
            target_pi = eps_greedy_probs_all(Q, TARGET_EPS)
            G, W = 0.0, 1.0
            for t in range(n_new - 1, -1, -1):
                s, a, r = data["s"][t], data["a"][t], data["r"][t]
                G = r + gamma * G
                C[s, a] += W
                Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
                W *= target_pi[s, a] / data["b"][t]
                if W == 0.0:
                    # the target policy would never have taken this action, so
                    # nothing earlier in the trajectory carries any weight
                    break
        else:
            G = 0.0
            for t in range(n_new - 1, -1, -1):
                s, a, r = data["s"][t], data["a"][t], data["r"][t]
                G = r + gamma * G
                counts[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / counts[s, a]

        # decay epsilon in proportion to the interaction actually consumed, so
        # that the schedule does not depend on episodes_per_iteration
        eps = max(eps - decay / max_steps * n_new, EPS_MIN)

        # Bellman error of the current Q against q_pi of the policy this run
        # is actually estimating, computed exactly from the true model.
        # On-policy that is the behaviour policy with the current epsilon;
        # off-policy it is the *target* policy, which is a different policy.
        # Scoring both against the behaviour policy would measure the
        # off-policy estimator against something it is not trying to
        # estimate, and make the two columns incomparable.
        pi = eps_greedy_probs_all(Q, TARGET_EPS if off_policy else eps)
        be = bellman_error(Q, solve_q_exactly(pi, mdp, gamma))
        greedy_v0 = solve_v_exactly(greedy_policy_from_q(Q), mdp, gamma)[0]

        lo, hi = tot_steps, min(tot_steps + n_new, max_steps)
        be_per_step[lo:hi] = be
        greedy_v0_per_step[lo:hi] = greedy_v0
        tot_steps += n_new

    return Q, be_per_step, greedy_v0_per_step


def sweep(off_policy: bool, reward_noise_std: float, mdp):
    """Run the full episodes x decays x seeds sweep for one configuration."""
    env = make_env(ENV_ID, max_episode_steps=HORIZON, reward_noise_std=reward_noise_std)
    n_states, n_actions = env.observation_space.n, env.action_space.n
    out = np.zeros((len(EPISODES_PER_ITERATION), len(DECAYS), N_SEEDS, MAX_STEPS))
    gv = np.zeros_like(out)
    for j, episodes in enumerate(EPISODES_PER_ITERATION):
        for k, decay in enumerate(DECAYS):
            for seed in range(N_SEEDS):
                Q = np.zeros((n_states, n_actions))
                _, be, g0 = monte_carlo(env, mdp, Q, GAMMA, decay, MAX_STEPS,
                                        episodes, off_policy, seed)
                out[j, k, seed], gv[j, k, seed] = be, g0
            print(f"    episodes {episodes:>2}  decay {decay}  "
                  f"final BE {np.nanmean(out[j, k, :, -1]):8.3f}   "
                  f"greedy v(s0) {np.nanmean(gv[j, k, :, -1]):+.4f} "
                  f"of {GAMMA ** 6:.4f}")
    env.close()
    return out, gv


def plot_pair(by_noise, gv_by_noise, title, out_path):
    """Top row: the Bellman error.  Bottom row: the fixed reference metric."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for col, noise in enumerate(sorted(by_noise)):
        for row, (res, ylab) in enumerate(
            [(by_noise[noise], "total absolute Bellman error"),
             (gv_by_noise[noise], "exact $v$ of the greedy policy at $s_0$")]
        ):
            ax = axs[row][col]
            ax.set_prop_cycle(color=CYCLE)
            for j, episodes in enumerate(EPISODES_PER_ITERATION):
                for k, decay in enumerate(DECAYS):
                    error_shade_plot(ax, res[j, k], stepsize=1,
                                     label=f"episodes {episodes}, decay {decay}")
            if row == 1:
                ax.axhline(GAMMA ** 6, color="k", ls=":", lw=1,
                           label=f"optimum {GAMMA ** 6:.3f}")
                ax.set_xlabel("environment steps")
                # the first few updates dip to about -9, which is the greedy
                # policy walking onto a penalty tile.  Letting that set the
                # axis hides the whole point of the panel, which is that the
                # curves sit on zero and never approach the dotted optimum.
                ax.set_ylim(-1.2, 0.7)
            ax.set_title(f"reward noise std = {noise}")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylab)
    axs[0][0].legend(fontsize=8, ncol=1)
    fig.suptitle(title, fontsize=13)
    save_fig(fig, out_path)


def main() -> None:
    mdp = extract_mdp(ENV_ID)
    rows = []
    store = {}

    for off_policy in (False, True):
        label = "off-policy weighted IS" if off_policy else "on-policy"
        by_noise, gv_by_noise = {}, {}
        for noise in (0.0, 3.0):
            print(f"  {label}, reward noise {noise}")
            by_noise[noise], gv_by_noise[noise] = sweep(off_policy, noise, mdp)
            for j, episodes in enumerate(EPISODES_PER_ITERATION):
                for k, decay in enumerate(DECAYS):
                    rows.append(dict(
                        method=label, reward_noise_std=noise,
                        episodes_per_iteration=episodes, decay=decay,
                        final_bellman_error=float(np.nanmean(by_noise[noise][j, k, :, -1])),
                        final_bellman_error_std=float(np.nanstd(by_noise[noise][j, k, :, -1])),
                        final_greedy_v0=float(np.nanmean(gv_by_noise[noise][j, k, :, -1])),
                        optimal_v0=float(GAMMA ** 6),
                    ))
        store[off_policy] = by_noise
        tag = "off_policy" if off_policy else "on_policy"
        (HERE / "results").mkdir(exist_ok=True)
        for noise in by_noise:
            np.save(HERE / "results" / f"{tag}_noise{noise}_be.npy",
                    by_noise[noise][..., ::10].astype(np.float32))
            np.save(HERE / "results" / f"{tag}_noise{noise}_greedy_v0.npy",
                    gv_by_noise[noise][..., ::10].astype(np.float32))
        plot_pair(by_noise, gv_by_noise,
                  f"Monte Carlo control, {label}",
                  HERE / "figures" / ("mc_importance.png" if off_policy else "mc_on_policy.png"))

    # head to head at one fixed setting so the two are directly comparable
    j, k = EPISODES_PER_ITERATION.index(10), DECAYS.index(1)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, noise in zip(axs, (0.0, 3.0)):
        error_shade_plot(ax, store[False][noise][j, k], label="on-policy", color=CYCLE[0])
        error_shade_plot(ax, store[True][noise][j, k], label="off-policy weighted IS",
                         color=CYCLE[2])
        ax.set_title(f"reward noise std = {noise}")
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
        ax.legend()
    axs[0].set_ylabel("total absolute Bellman error")
    fig.suptitle("On-policy against off-policy, 10 episodes per update, decay 1", fontsize=13)
    save_fig(fig, HERE / "figures" / "mc_on_vs_off.png")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / "final_bellman_error.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out/'final_bellman_error.csv'}")


if __name__ == "__main__":
    main()
