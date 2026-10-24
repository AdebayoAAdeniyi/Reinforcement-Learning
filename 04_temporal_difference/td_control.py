"""Temporal difference control: Q-learning, SARSA, Expected SARSA, and doubles.

Task
----
Run Q-learning, SARSA and Expected SARSA on `Penalty-3x3-v0` from three
initial Q values (-10, 0, 10), with and without reward noise of std 3.0, and
then repeat with the double estimator versions.  Track the TD error at every
step, and the Bellman error and expected return every hundred steps.

Reference
---------
Sutton and Barto, 2nd edition.  SARSA is section 6.4 and equation 6.7,
Q-learning is section 6.5 and equation 6.8, Expected SARSA is section 6.6 and
equation 6.9, Double Q-learning is section 6.7 and equation 6.10.

Double SARSA and Double Expected SARSA are not in the book.  They are built
here by the same construction as Double Q-learning, holding two independent
estimates and updating one of them using the other to supply the value, and
they are labelled as extensions rather than as book algorithms.

Bootstrapping at the end of an episode
--------------------------------------
`terminated` and `truncated` mean different things and must be treated
differently.  `terminated` says the MDP itself ended, so there is no next
state and the target is just the reward.  `truncated` says the time limit cut
the episode off while the MDP was still running, so the target must still
bootstrap, otherwise the agent is taught that the world ends after ten steps.
Collapsing them into one `done` flag and bootstrapping in both cases puts a
spurious `gamma * Q(s_next, .)` on the one genuinely terminal transition in
this environment, which is the only place the +1 reward enters.

Outputs
-------
figures/td_noise_0.0.png, figures/td_noise_3.0.png      single estimators
figures/double_noise_0.0.png, figures/double_noise_3.0.png   double estimators
figures/single_vs_double.png                            head to head
results/td_summary.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit.bellman import bellman_error, solve_q_exactly  # noqa: E402
from rlkit.envs import extract_mdp, make_env, reset_seeded  # noqa: E402
from rlkit.evaluation import expected_return  # noqa: E402
from rlkit.plotting import CYCLE, error_shade_plot, save_fig  # noqa: E402
from rlkit.policies import eps_greedy_probs, eps_greedy_probs_all  # noqa: E402

HERE = Path(__file__).resolve().parent
ENV_ID = "Gym-Gridworlds/Penalty-3x3-v0"
GAMMA = 0.99
ALPHA0 = 0.1
EPS0 = 1.0
EPS_MIN, ALPHA_MIN = 0.01, 0.001
MAX_STEPS = 10_000
LOG_EVERY = 100
HORIZON = 10
INIT_VALUES = [-10.0, 0.0, 10.0]
SINGLE_ALGS = ["QL", "SARSA", "ExpSARSA"]
DOUBLE_ALGS = ["DoubleQL", "DoubleSARSA", "DoubleExpSARSA"]
N_SEEDS = 10
EVAL_EPISODES = 10


def td_target(alg, Q_value, Q_behaviour, s_next, a_next, eps):
    """The bootstrapped part of the target, before reward and discount.

    `Q_behaviour` chooses which action or distribution to use, `Q_value`
    supplies the number.  For the single estimator algorithms they are the
    same array; for the double ones they are the two different estimates,
    which is exactly the decoupling that removes the maximisation bias.
    """
    if alg.endswith("QL"):
        return Q_value[s_next, int(np.argmax(Q_behaviour[s_next]))]
    if alg.endswith("SARSA") and not alg.endswith("ExpSARSA"):
        return Q_value[s_next, a_next]
    probs = eps_greedy_probs(Q_behaviour[s_next], eps)
    return float(np.dot(probs, Q_value[s_next]))


def td_control(env, env_eval, mdp, alg, init_value, seed, reward_noise_std):
    """One training run.  Returns TD errors, Bellman errors and returns."""
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.observation_space.n, env.action_space.n
    double = alg.startswith("Double")

    Q1 = np.full((n_states, n_actions), float(init_value))
    Q2 = np.full((n_states, n_actions), float(init_value)) if double else None

    def combined():
        """The estimate used for acting and for reporting."""
        return (Q1 + Q2) / 2.0 if double else Q1

    eps, alpha = EPS0, ALPHA0
    eps_decay, alpha_decay = EPS0 / MAX_STEPS, ALPHA0 / MAX_STEPS

    tde = np.zeros(MAX_STEPS)
    be, ret = [], []
    # state visitation, kept because the effect of the initial value on this
    # problem turns out to be an exploration effect, and a visit histogram is
    # the evidence for that rather than an assertion about it
    visits = np.zeros(n_states)

    s, _ = reset_seeded(env, seed)
    a = int(rng.choice(n_actions, p=eps_greedy_probs(combined()[s], eps)))

    for step in range(MAX_STEPS):
        s_next, r, terminated, truncated, _ = env.step(a)

        # The next action is needed by SARSA and is also the action actually
        # taken next, so it is drawn once here and reused below rather than
        # being redrawn per algorithm.  Drawing it in only some branches is
        # how an agent ends up repeating one action for a whole episode.
        a_next = int(rng.choice(n_actions, p=eps_greedy_probs(combined()[s_next], eps)))

        if double:
            # update one estimate chosen at random, valued by the other
            if rng.random() < 0.5:
                Q_update, Q_value = Q1, Q2
            else:
                Q_update, Q_value = Q2, Q1
            boot = td_target(alg, Q_value, combined(), s_next, a_next, eps)
        else:
            Q_update = Q1
            boot = td_target(alg, Q1, Q1, s_next, a_next, eps)

        # `terminated` removes the bootstrap, `truncated` does not
        target = r + (0.0 if terminated else GAMMA * boot)
        delta = target - Q_update[s, a]
        Q_update[s, a] += alpha * delta
        tde[step] = abs(delta)
        visits[s] += 1

        if (step + 1) % LOG_EVERY == 0:
            Q = combined()
            pi = eps_greedy_probs_all(Q, eps)
            be.append(bellman_error(Q, solve_q_exactly(pi, mdp, GAMMA)))
            ret.append(expected_return(env_eval, Q, GAMMA, EVAL_EPISODES))

        if terminated or truncated:
            s, _ = env.reset()
            a = int(rng.choice(n_actions, p=eps_greedy_probs(combined()[s], eps)))
        else:
            s, a = s_next, a_next

        eps = max(eps - eps_decay, EPS_MIN)
        alpha = max(alpha - alpha_decay, ALPHA_MIN)

    return tde, np.array(be), np.array(ret), visits


def run_group(algs, reward_noise_std, mdp):
    env = make_env(ENV_ID, max_episode_steps=HORIZON, reward_noise_std=reward_noise_std)
    env_eval = make_env(ENV_ID, max_episode_steps=HORIZON)
    n_log = MAX_STEPS // LOG_EVERY
    tde = np.zeros((len(INIT_VALUES), len(algs), N_SEEDS, MAX_STEPS))
    be = np.zeros((len(INIT_VALUES), len(algs), N_SEEDS, n_log))
    ret = np.zeros((len(INIT_VALUES), len(algs), N_SEEDS, n_log))
    n_states = env.observation_space.n
    vis = np.zeros((len(INIT_VALUES), len(algs), N_SEEDS, n_states))
    for i, iv in enumerate(INIT_VALUES):
        for j, alg in enumerate(algs):
            for seed in range(N_SEEDS):
                t, b, g, v = td_control(env, env_eval, mdp, alg, iv, seed, reward_noise_std)
                tde[i, j, seed], be[i, j, seed] = t, b
                ret[i, j, seed], vis[i, j, seed] = g, v
            frac = vis[i, j].mean(0)
            frac = frac / frac.sum()
            print(f"    Q0={iv:>5.0f}  {alg:<14}  final BE {be[i, j, :, -1].mean():7.3f}  "
                  f"final return {ret[i, j, :, -1].mean():+.3f}  "
                  f"largest share of visits in one state {frac.max():.2f}")
    env.close()
    env_eval.close()
    return tde, be, ret, vis


def plot_group(tde, be, ret, algs, title, out_path):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax in axs:
        ax.set_prop_cycle(color=CYCLE)
        ax.set_xlabel("steps")
        ax.grid(alpha=0.3)
    for i, iv in enumerate(INIT_VALUES):
        for j, alg in enumerate(algs):
            label = f"$Q_0$={iv:g}, {alg}"
            error_shade_plot(axs[0], tde[i, j], stepsize=1, smoothing_window=50, label=label)
            error_shade_plot(axs[1], be[i, j], stepsize=LOG_EVERY, smoothing_window=5, label=label)
            error_shade_plot(axs[2], ret[i, j], stepsize=LOG_EVERY, smoothing_window=5, label=label)
    axs[0].set_title("TD error $|\\delta_t|$")
    axs[0].set_ylim(0, 5)
    axs[1].set_title("Bellman error against $q_\\pi$")
    axs[1].set_ylim(0, 60)
    axs[2].set_title("expected return of the greedy policy")
    axs[2].set_ylim(-5, 1.1)
    axs[2].legend(fontsize=7, loc="lower right", ncol=1)
    fig.suptitle(title, fontsize=13)
    save_fig(fig, out_path)


def main() -> None:
    mdp = extract_mdp(ENV_ID)
    rows, store = [], {}
    for algs, tag in ((SINGLE_ALGS, "td"), (DOUBLE_ALGS, "double")):
        for noise in (0.0, 3.0):
            print(f"  {tag}, reward noise {noise}")
            tde, be, ret, vis = run_group(algs, noise, mdp)
            store[(tag, noise)] = (tde, be, ret, vis)
            plot_group(tde, be, ret, algs,
                       f"{'Single' if tag == 'td' else 'Double'} estimators, "
                       f"reward noise std = {noise}",
                       HERE / "figures" / f"{tag}_noise_{noise}.png")
            for i, iv in enumerate(INIT_VALUES):
                for j, alg in enumerate(algs):
                    rows.append(dict(
                        algorithm=alg, reward_noise_std=noise, init_value=iv,
                        final_bellman_error=float(be[i, j, :, -1].mean()),
                        final_bellman_error_std=float(be[i, j, :, -1].std()),
                        final_return=float(ret[i, j, :, -1].mean()),
                        final_return_std=float(ret[i, j, :, -1].std()),
                        mean_td_error_last_1000=float(tde[i, j, :, -1000:].mean()),
                        max_state_visit_share=float(
                            (vis[i, j].mean(0) / vis[i, j].mean(0).sum()).max()),
                    ))

    # head to head: each single algorithm against its double, at Q0 = 0
    i = INIT_VALUES.index(0.0)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for col, noise in enumerate((0.0, 3.0)):
        for row, (metric, idx, ylab) in enumerate(
            [("Bellman error", 1, "Bellman error"), ("return", 2, "expected return")]
        ):
            ax = axs[row][col]
            for j in range(3):
                error_shade_plot(ax, store[("td", noise)][idx][i, j], stepsize=LOG_EVERY,
                                 smoothing_window=5, label=SINGLE_ALGS[j], color=CYCLE[j])
                error_shade_plot(ax, store[("double", noise)][idx][i, j], stepsize=LOG_EVERY,
                                 smoothing_window=5, label=DOUBLE_ALGS[j],
                                 color=CYCLE[j], linestyle="--")
            ax.set_title(f"{metric}, reward noise {noise}")
            ax.set_xlabel("steps")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Single against double estimators, $Q_0 = 0$ (dashed = double)", fontsize=13)
    save_fig(fig, HERE / "figures" / "single_vs_double.png")

    # where the agent actually spent its time, which explains the effect of
    # the initial value better than the learning curves do
    fig, axs = plt.subplots(1, len(INIT_VALUES), figsize=(4.0 * len(INIT_VALUES), 3.8))
    for c, iv in enumerate(INIT_VALUES):
        v = store[("td", 0.0)][3][c, SINGLE_ALGS.index("QL")].mean(0)
        v = v / v.sum()
        im = axs[c].imshow(v.reshape(3, 3), cmap="magma", vmin=0, vmax=v.max())
        axs[c].set_title(f"Q-learning, $Q_0$ = {iv:g}", fontsize=10)
        axs[c].set_xticks([])
        axs[c].set_yticks([])
        for s in range(9):
            axs[c].text(s % 3, s // 3, f"{v[s]:.2f}", ha="center", va="center",
                        color="w", fontsize=9)
        fig.colorbar(im, ax=axs[c], fraction=0.046)
    fig.suptitle("Share of training steps spent in each state, averaged over 10 seeds",
                 fontsize=13)
    save_fig(fig, HERE / "figures" / "state_visitation.png")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / "td_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out/'td_summary.csv'}")


if __name__ == "__main__":
    main()
