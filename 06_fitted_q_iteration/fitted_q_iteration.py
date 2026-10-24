"""Fitted Q-Iteration with a linear approximator, and the N / K / D sweep.

Task
----
Implement FQI on `Gym-Gridworlds/Empty-2x2-v0` with state aggregation
features, and sweep the three hyperparameters over {1, 100, 1000}:

    N   gradient steps taken per fit, with the target held fixed
    K   fitting iterations, each of which recomputes the target
    D   number of transitions in the dataset used for a fit

Reference
---------
Ernst, Geurts and Wehenkel, *Tree-Based Batch Mode Reinforcement Learning*,
JMLR 2005, is the original FQI paper.  The connection to the book is
Sutton and Barto section 9.3 for the semi-gradient update and section 11.3
for why fixing the target matters.

The algorithm, and the two conditions it is trying to restore
-------------------------------------------------------------
Regression assumes the data is independent and identically distributed and
that the target is stationary.  Online Q-learning violates both: consecutive
transitions come from one trajectory, and the target `r + gamma max_a' Q(s',a')`
moves every time the weights move.  FQI addresses them separately.

    repeat K times:
        compute the target for the whole dataset, using the current weights
        repeat N times:
            one gradient step on the squared error against that fixed target

Holding the target fixed for the N inner steps makes the inner loop an
ordinary stationary regression problem.  Using a batch rather than the single
most recent transition is what addresses, or fails to address, independence.

Two data collection modes are provided, because the assignment questions are
about the difference between them:

`on_the_fly`  collect D fresh consecutive transitions, then fit on exactly
              those.  The D transitions come from one stretch of one
              trajectory, so they are strongly correlated and the i.i.d.
              condition is broken.
`replay`      keep a buffer of the last `BUFFER_SIZE` transitions, far more
              than D, and sample D of them uniformly at random for each fit.
              Sampling across a long history decorrelates the batch and
              restores the condition approximately.

The buffer has to be much larger than D for this comparison to mean
anything.  If the buffer held only the last D transitions, sampling D of them
would draw from exactly the same window the `on_the_fly` batch comes from,
and the two modes would differ only in the multiplicity of repeated samples.
That is a bootstrap resample of one correlated window, not an independence
test.  `BUFFER_SIZE` is therefore the whole run.

The two modes then differ at every D, including D = 1, where `on_the_fly`
fits the transition that just happened and `replay` fits one transition drawn
uniformly from the whole history.  Those are genuinely different algorithms
and the results below show it, so D = 1 is not a control condition.

Running both settles the question by measurement rather than asserting an
answer, and the size of the effect turns out to depend sharply on how many
gradient updates are performed.

N = K = D = 1 is online Q-learning with a linear approximator: one
transition, one target, one gradient step.

Outputs
-------
figures/fqi_sweep_on_the_fly.png   return and TD error for all 27 settings
figures/fqi_sweep_replay.png       the same with the replay buffer
figures/fqi_iid_comparison.png     the two modes head to head
results/fqi_summary.csv
"""

from __future__ import annotations

import csv
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit.envs import make_env, reset_seeded  # noqa: E402
from rlkit.evaluation import expected_return_linear  # noqa: E402
from rlkit.features import aggregation_features, make_centers  # noqa: E402
from rlkit.plotting import save_fig  # noqa: E402

HERE = Path(__file__).resolve().parent
ENV_ID = "Gym-Gridworlds/Empty-2x2-v0"
GAMMA = 0.99
ALPHA = 0.05
MAX_STEPS = 10_000
LOG_EVERY = 100
EPS_FLOOR = 0.5
N_SEEDS = 5
EVAL_EPISODES = 10
SWEEP = [1, 100, 1000]
TOL = 1e-5
#: Size of the replay buffer, deliberately the whole run rather than D.
BUFFER_SIZE = MAX_STEPS


def fqi(env, env_eval, get_phi, n_features, n_actions,
        gradient_steps, fitting_iterations, datasize, mode, seed):
    """One FQI run.  Returns the logged return and mean absolute TD error."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_features, n_actions))
    buffer = deque(maxlen=BUFFER_SIZE if mode == "replay" else max(datasize, 1))
    eps = 1.0
    td_error = np.nan

    n_log = MAX_STEPS // LOG_EVERY
    ret_log = np.zeros(n_log)
    tde_log = np.zeros(n_log)
    log_i = 0

    s, _ = reset_seeded(env, seed)
    for step in range(MAX_STEPS):
        phi = get_phi(np.asarray(s, dtype=float))
        q = (phi @ weights).ravel()
        a = (int(rng.integers(n_actions)) if rng.random() < eps
             else int(np.argmax(q)))
        s_next, r, terminated, truncated, _ = env.step(a)
        buffer.append((np.asarray(s, dtype=float), a, r,
                       np.asarray(s_next, dtype=float), float(terminated)))

        # an update routine happens once every `datasize` steps, so that each
        # transition is collected once and used once.  Running it on every
        # step instead would reuse the same batch D times over and make the
        # effective number of updates depend on D in a second, hidden way.
        if len(buffer) >= datasize and (step + 1) % datasize == 0:
            if mode == "replay":
                idx = rng.integers(len(buffer), size=datasize)
                batch = [buffer[i] for i in idx]
            else:
                batch = list(buffer)[-datasize:]

            S = np.stack([b[0] for b in batch])
            A = np.array([b[1] for b in batch])
            R = np.array([b[2] for b in batch])
            S_next = np.stack([b[3] for b in batch])
            TERM = np.array([b[4] for b in batch])
            Phi, Phi_next = get_phi(S), get_phi(S_next)
            rows = np.arange(datasize)
            masks = [A == act for act in range(n_actions)]

            for _k in range(fitting_iterations):
                w_before_fit = weights.copy()
                # the target is computed once and then held fixed for the
                # whole inner loop: this is what makes the inner problem a
                # stationary regression
                target = R + GAMMA * (1.0 - TERM) * (Phi_next @ weights).max(axis=1)

                abs_td = np.zeros(datasize)
                for _n in range(gradient_steps):
                    w_before_step = weights.copy()
                    pred = (Phi @ weights)[rows, A]
                    delta = target - pred
                    abs_td = np.abs(delta)
                    for act, m in enumerate(masks):
                        if m.any():
                            weights[:, act] += ALPHA * (Phi[m].T @ delta[m]) / m.sum()
                    if np.allclose(weights, w_before_step, rtol=TOL, atol=TOL):
                        break
                td_error = float(abs_td.mean())
                if np.allclose(weights, w_before_fit, rtol=TOL, atol=TOL):
                    break

        if (step + 1) % LOG_EVERY == 0:
            ret_log[log_i] = expected_return_linear(
                env_eval, weights, GAMMA, get_phi, EVAL_EPISODES)
            tde_log[log_i] = td_error
            log_i += 1

        s = s_next if not (terminated or truncated) else env.reset()[0]
        eps = max(eps - 1.0 / MAX_STEPS, EPS_FLOOR)

    return ret_log, tde_log


def setup():
    env = make_env(ENV_ID, coordinate_observation=True,
                   random_action_prob=0.1, reward_noise_std=0.01)
    env_eval = make_env(ENV_ID, coordinate_observation=True, max_episode_steps=10)
    centers = make_centers(env.observation_space.low, env.observation_space.high, [2, 2])
    state_dim = env.observation_space.shape[0]

    def get_phi(state):
        return aggregation_features(np.asarray(state, float).reshape(-1, state_dim), centers)

    n_features = get_phi(env.reset()[0]).shape[1]
    return env, env_eval, get_phi, n_features, env.action_space.n


def run_sweep(mode: str, rows: list[dict]):
    env, env_eval, get_phi, n_features, n_actions = setup()
    n_log = MAX_STEPS // LOG_EVERY
    ret = np.zeros((len(SWEEP), len(SWEEP), len(SWEEP), N_SEEDS, n_log))
    tde = np.zeros_like(ret)
    for i, N in enumerate(SWEEP):
        for j, K in enumerate(SWEEP):
            for k, D in enumerate(SWEEP):
                for seed in range(N_SEEDS):
                    r, t = fqi(env, env_eval, get_phi, n_features, n_actions,
                               N, K, D, mode, seed)
                    ret[i, j, k, seed], tde[i, j, k, seed] = r, t
                final_ret = ret[i, j, k, :, -1].mean()
                final_tde = np.nanmean(tde[i, j, k, :, -1])
                rows.append(dict(mode=mode, N=N, K=K, D=D,
                                 final_return=float(final_ret),
                                 final_return_std=float(ret[i, j, k, :, -1].std()),
                                 final_td_error=float(final_tde)))
                print(f"    N={N:<5} K={K:<5} D={D:<5}  "
                      f"return {final_ret:+.3f}  TD error {final_tde:.4f}")
    env.close()
    env_eval.close()
    return ret, tde


def plot_sweep(ret, tde, mode, out_path):
    fig, axs = plt.subplots(6, 9, figsize=(22, 13), sharex=True)
    for i, N in enumerate(SWEEP):
        for j, K in enumerate(SWEEP):
            for k, D in enumerate(SWEEP):
                c = k + i * 3
                x = np.arange(ret.shape[-1]) * LOG_EVERY
                axs[j][c].plot(x, ret[i, j, k].mean(0), color="#2ca02c")
                axs[j][c].set_ylim(0, 1.1)
                axs[j][c].set_title(f"N={N} K={K} D={D}", fontsize=7)
                axs[j][c].tick_params(labelsize=6)
                axs[j][c].grid(alpha=0.3)
                axs[j + 3][c].plot(x, np.nanmean(tde[i, j, k], axis=0), color="#1f77b4")
                axs[j + 3][c].tick_params(labelsize=6)
                axs[j + 3][c].grid(alpha=0.3)
                if c == 0:
                    axs[j][c].set_ylabel("return", fontsize=8)
                    axs[j + 3][c].set_ylabel("TD error", fontsize=8)
                if j == 2:
                    axs[j + 3][c].set_xlabel("steps", fontsize=7)
    fig.suptitle(f"Fitted Q-Iteration, {mode.replace('_', ' ')} batches.  "
                 f"Top three rows: expected return.  Bottom three: mean absolute TD error.",
                 fontsize=13)
    save_fig(fig, out_path)


def plot_iid_comparison(on_the_fly, replay, d_index: int = 1):
    """Consecutive batches against uniformly sampled ones, across N and K.

    Fixed at D = 100 by default, because that is where the task is hard
    enough to discriminate: at D = 1000 every setting reaches the optimum
    under both modes, and at D = 1 the two modes are different algorithms
    rather than two ways of batching (see the module docstring).
    """
    D = SWEEP[d_index]
    fig, axs = plt.subplots(len(SWEEP), len(SWEEP), figsize=(13, 9.5),
                            sharex=True, sharey=True)
    x = np.arange(on_the_fly.shape[-1]) * LOG_EVERY
    for i, N in enumerate(SWEEP):
        for j, K in enumerate(SWEEP):
            ax = axs[j][i]
            a = on_the_fly[i, j, d_index].mean(0)
            b = replay[i, j, d_index].mean(0)
            ax.plot(x, a, color="#d62728", label="consecutive batch")
            ax.plot(x, b, color="#1f77b4", label="sampled from buffer")
            ax.set_title(f"N={N}, K={K}   final {a[-1]:.2f} vs {b[-1]:.2f}",
                         fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.grid(alpha=0.3)
            if j == len(SWEEP) - 1:
                ax.set_xlabel("steps")
            if i == 0:
                ax.set_ylabel("expected return")
    axs[0][0].legend(fontsize=8)
    fig.suptitle(f"Consecutive batches against uniformly sampled batches, D = {D}",
                 fontsize=13)
    save_fig(fig, HERE / "figures" / "fqi_iid_comparison.png")


def main() -> None:
    rows: list[dict] = []
    store = {}
    for mode in ("on_the_fly", "replay"):
        print(f"  mode: {mode}")
        ret, tde = run_sweep(mode, rows)
        store[mode] = (ret, tde)
        out = HERE / "results"
        out.mkdir(exist_ok=True)
        np.save(out / f"{mode}_return.npy", ret)
        np.save(out / f"{mode}_td_error.npy", tde)
        plot_sweep(ret, tde, mode, HERE / "figures" / f"fqi_sweep_{mode}.png")

    plot_iid_comparison(store["on_the_fly"][0], store["replay"][0])

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / "fqi_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out/'fqi_summary.csv'}")


if __name__ == "__main__":
    main()
