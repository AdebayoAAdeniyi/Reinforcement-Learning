"""Actor-critic: n-step returns and TD(lambda) against the Monte Carlo baseline.

What this chapter is about
--------------------------
REINFORCE in chapter 07 weights each log gradient by the full Monte Carlo
return `G_t`, which is unbiased and high variance.  An actor-critic replaces
that with a learned value function: the actor is the policy, the critic is a
second linear approximator that estimates `v_pi`, and the weight on each log
gradient becomes an advantage built from the critic.  Trading some bias for a
large reduction in variance is the whole idea.

Three critics are compared, which is exactly the bias and variance axis:

`mc`        the full return.  No bias, most variance.  This is REINFORCE with
            a learned state value baseline, included as the reference point.
`nstep`     the n-step return, Sutton and Barto section 7.1 and equation 7.1.
            `n` controls the trade-off directly: n = 1 is maximum bootstrap
            and minimum variance, large n approaches Monte Carlo.
`lambda`    the lambda-return, section 12.1 and equation 12.2, computed
            backwards from the TD errors.  A geometric average over all
            n-step returns, so it blends every horizon at once rather than
            picking one.

Reference
---------
Sutton and Barto, 2nd edition.  Actor-critic with eligibility traces is the
box in section 13.5 and the one-step version is equation 13.17.  The n-step
return is equation 7.1, the lambda-return is equation 12.2, and the forward
view used here is section 12.1.

Structure
---------
The three critics differ in exactly one function, `advantages`, which turns a
trajectory into per step advantages and critic targets.  Everything else, the
rollout, the actor update and the critic update, is shared, so a difference
in the curves is a difference in the critic and not in anything else.

Note the bootstrap convention in `advantages`: `V[T]` is zero when the
episode terminated and the critic's own estimate when it was truncated by the
time limit.  That distinction is the whole of the critic's contract with the
environment and is the easiest thing in this chapter to get wrong.

The environment id is also corrected: `DangerMaze-6x6-v0` exists in the
October 2024 `gym_gridworlds` pinned in `requirements.txt` and was removed
upstream later, so `rlkit.envs.make_env` resolves it and says so clearly if
it cannot.

Outputs
-------
figures/actor_critic_comparison.png   the three critics
figures/actor_critic_nstep.png        the effect of n
results/actor_critic_summary.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit.envs import make_env, reset_seeded  # noqa: E402
from rlkit.evaluation import expected_return_linear  # noqa: E402
from rlkit.features import make_centers, rbf_features  # noqa: E402
from rlkit.plotting import CYCLE, error_shade_plot, save_fig  # noqa: E402
from rlkit.policies import dlog_softmax_probs, softmax_probs  # noqa: E402

HERE = Path(__file__).resolve().parent
# The comparison between critics runs on the same small gridworld used by
# chapters 01 to 04 and 07, so that the differences in the curves are
# differences between the critics and not between environments.
# `DangerMaze-6x6-v0` is reported separately at the end as a harder case.
ENV_ID = "Gym-Gridworlds/Penalty-3x3-v0"
HARD_ENV_ID = "Gym-Gridworlds/DangerMaze-6x6-v0"
GAMMA = 0.99
ALPHA_ACTOR = 0.01
ALPHA_CRITIC = 0.05
TEMPERATURE = 1.0
EPISODES_PER_UPDATE = 10
MAX_STEPS = 150_000
N_CENTERS = 7
N_SEEDS = 5
EPISODES_EVAL = 5
TRAIN_KWARGS = dict(coordinate_observation=True, distance_reward=True,
                    max_episode_steps=50)
EVAL_KWARGS = dict(coordinate_observation=True, max_episode_steps=10)
HARD_EVAL_KWARGS = dict(coordinate_observation=True, max_episode_steps=30)


def build_features(env):
    dim = env.observation_space.shape[0]
    low, high = env.observation_space.low, env.observation_space.high
    centers = make_centers(low, high, [N_CENTERS] * dim, pad=0.1)
    sigmas = (np.asarray(high) - np.asarray(low)) / N_CENTERS * 0.75 + 1e-8

    def get_phi(state):
        return rbf_features(np.asarray(state, float).reshape(-1, dim), centers, sigmas)
    return get_phi, get_phi(env.reset()[0]).shape[1]


def rollout(env, actor, get_phi, rng, seed):
    """One episode under the current softmax policy."""
    s, _ = reset_seeded(env, seed)
    phis, acts, rews = [], [], []
    terminated = False
    while True:
        phi = get_phi(s)
        probs = softmax_probs(phi, actor, TEMPERATURE).ravel()
        a = int(rng.choice(probs.shape[0], p=probs))
        s, r, terminated, truncated, _ = env.step(a)
        phis.append(phi)
        acts.append(a)
        rews.append(float(r))
        if terminated or truncated:
            break
    return phis, np.array(acts), np.array(rews), terminated


def advantages(critic_kind, phis, rews, terminated, critic_w, n_step, lam):
    """Return the per step advantage and the per step critic target.

    All three share the same bootstrap convention.  `V[T]` is the value of the
    state the episode ended in, which is zero when the episode terminated and
    the critic's estimate when it was truncated by the time limit.  Treating a
    truncation as a termination is the most common silent error in this part
    of the course, and it teaches the agent that the world ends after
    `max_episode_steps`.
    """
    T = len(rews)
    V = np.array([float(np.asarray(phi @ critic_w).ravel()[0]) for phi in phis] + [0.0])
    if not terminated:
        # the episode was truncated, so bootstrap from the last state
        V[T] = float(np.asarray(phis[-1] @ critic_w).ravel()[0])

    if critic_kind == "mc":
        G = np.zeros(T)
        running = 0.0 if terminated else V[T]
        for t in range(T - 1, -1, -1):
            running = rews[t] + GAMMA * running
            G[t] = running
        return G - V[:T], G

    if critic_kind == "nstep":
        G = np.zeros(T)
        for t in range(T):
            end = min(t + n_step, T)
            g = 0.0
            for k in range(end - 1, t - 1, -1):
                g = rews[k] + GAMMA * g
            g += (GAMMA ** (end - t)) * V[end]
            G[t] = g
        return G - V[:T], G

    if critic_kind == "lambda":
        # backward accumulation of the lambda-return, equation 12.2 in its
        # recursive form: G_lambda(t) = r + gamma[(1-lam) V(s') + lam G_lambda(t+1)]
        G = np.zeros(T)
        running = V[T]
        for t in range(T - 1, -1, -1):
            running = rews[t] + GAMMA * ((1 - lam) * V[t + 1] + lam * running)
            G[t] = running
        return G - V[:T], G

    raise ValueError(critic_kind)


def actor_critic(critic_kind, seed, n_step=5, lam=0.9,
                 env_id=None, eval_kwargs=None, max_steps=None,
                 report_train_reward=False):
    """One training run.  Returns the expected return logged per step."""
    env_id = env_id or ENV_ID
    eval_kwargs = eval_kwargs or EVAL_KWARGS
    max_steps = max_steps or MAX_STEPS
    env = make_env(env_id, **TRAIN_KWARGS)
    env_eval = make_env(env_id, **eval_kwargs)
    get_phi, n_features = build_features(env)
    rng = np.random.default_rng(seed)

    actor = np.zeros((n_features, env.action_space.n))
    critic = np.zeros(n_features)

    history = np.zeros(max_steps)
    train_reward = []
    tot_steps = 0
    seed_base = seed * 1_000_000

    while tot_steps < max_steps:
        actor_grad = np.zeros_like(actor)
        critic_grad = np.zeros_like(critic)
        n_episodes = 0
        steps_this_update = 0

        for e in range(EPISODES_PER_UPDATE):
            phis, acts, rews, terminated = rollout(env, actor, get_phi, rng,
                                                   seed_base + e)
            adv, target = advantages(critic_kind, phis, rews, terminated,
                                     critic, n_step, lam)
            for t, (phi, a) in enumerate(zip(phis, acts)):
                actor_grad += adv[t] * dlog_softmax_probs(phi, actor, TEMPERATURE, a)
                # semi-gradient critic update: the target is held constant
                critic_grad += (target[t] - float(np.asarray(phi @ critic).ravel()[0])) * phi.ravel()
            n_episodes += 1
            steps_this_update += len(rews)
            if report_train_reward:
                train_reward.append(float(rews.sum()))
        seed_base += EPISODES_PER_UPDATE

        actor = actor + ALPHA_ACTOR * actor_grad / n_episodes
        critic = critic + ALPHA_CRITIC * critic_grad / steps_this_update

        value = expected_return_linear(env_eval, actor, GAMMA, get_phi, EPISODES_EVAL)
        hi = min(tot_steps + steps_this_update, max_steps)
        history[tot_steps:hi] = value
        tot_steps += steps_this_update

    env.close()
    env_eval.close()
    if report_train_reward:
        return history, np.array(train_reward)
    return history


def hard_case(rows):
    """A harder environment, reported because none of the critics solves it.

    None of the three critics solves `DangerMaze-6x6-v0` within this budget.
    That is reported rather than quietly dropped, with the training reward
    alongside the evaluation return so the reader can see what does happen:
    the agent learns to stop falling into the pits, which takes the shaped
    training reward from about -100 to about -9, and then stops improving
    because the softmax policy has saturated and the log gradient has gone to
    zero.  Learning to avoid a -100 penalty and learning to reach a +1 goal
    are different problems, and the first one dominates the gradient.
    """
    print(f"  harder case: {HARD_ENV_ID}")
    steps = 60_000
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, kind in enumerate(["mc", "nstep", "lambda"]):
        hist, train = actor_critic(kind, 0, env_id=HARD_ENV_ID,
                                   eval_kwargs=HARD_EVAL_KWARGS,
                                   max_steps=steps, report_train_reward=True)
        rows.append(dict(variant=f"{kind} on {HARD_ENV_ID}",
                         final_return=float(hist[-1]), final_return_std=0.0,
                         mean_over_training=float(hist.mean())))
        print(f"    {kind:<8} eval return {hist[-1]:+7.3f}   "
              f"training reward per episode {train[0]:+8.2f} -> {train[-1]:+8.2f}")
        axs[0].plot(np.arange(steps), hist, label=kind, color=CYCLE[i])
        axs[1].plot(np.convolve(train, np.ones(20) / 20, mode="valid"),
                    label=kind, color=CYCLE[i])
    axs[0].set_xlabel("environment steps")
    axs[0].set_ylabel("evaluation return (sparse reward)")
    axs[0].set_title("none of the three reaches the goal")
    axs[1].set_xlabel("episode")
    axs[1].set_ylabel("training reward per episode (shaped)")
    axs[1].set_title("but all three learn to avoid the pits")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Actor-critic on {HARD_ENV_ID}, one seed", fontsize=13)
    save_fig(fig, HERE / "figures" / "actor_critic_dangermaze.png")


def main() -> None:
    rows = []
    (HERE / "results").mkdir(exist_ok=True)

    print("  three critics")
    labels = [("mc", "Monte Carlo return"),
              ("nstep", "5-step return"),
              ("lambda", "lambda-return, lambda = 0.9")]
    results = np.zeros((len(labels), N_SEEDS, MAX_STEPS))
    for i, (kind, label) in enumerate(labels):
        for seed in range(N_SEEDS):
            results[i, seed] = actor_critic(kind, seed)
        final = results[i, :, -1]
        rows.append(dict(variant=label, final_return=float(final.mean()),
                         final_return_std=float(final.std()),
                         mean_over_training=float(results[i].mean())))
        print(f"    {label:<30} final {final.mean():+7.3f} +/- {final.std():5.3f}   "
              f"mean over training {results[i].mean():+7.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, (_kind, label) in enumerate(labels):
        error_shade_plot(ax, results[i], stepsize=1, smoothing_window=50,
                         label=label, color=CYCLE[i])
    ax.set_xlabel("environment steps")
    ax.set_ylabel("expected return")
    ax.set_title(f"Actor-critic on {ENV_ID}, {N_SEEDS} seeds")
    # the first few updates dive to about -35 while the policy is still
    # random, which squashes the informative range into a flat line if the
    # axis is left automatic
    ax.set_ylim(-2.0, 1.1)
    ax.grid(alpha=0.3)
    ax.legend()
    np.save(HERE / "results" / "critic_curves.npy",
            results[..., ::100].astype(np.float32))
    save_fig(fig, HERE / "figures" / "actor_critic_comparison.png")

    print("  effect of n")
    n_values = [1, 3, 5, 20]
    res_n = np.zeros((len(n_values), N_SEEDS, MAX_STEPS))
    for i, n in enumerate(n_values):
        for seed in range(N_SEEDS):
            res_n[i, seed] = actor_critic("nstep", seed, n_step=n)
        final = res_n[i, :, -1]
        rows.append(dict(variant=f"{n}-step return", final_return=float(final.mean()),
                         final_return_std=float(final.std()),
                         mean_over_training=float(res_n[i].mean())))
        print(f"    n = {n:<3} final {final.mean():+7.3f} +/- {final.std():5.3f}   "
              f"mean over training {res_n[i].mean():+7.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, n in enumerate(n_values):
        error_shade_plot(ax, res_n[i], stepsize=1, smoothing_window=50,
                         label=f"n = {n}", color=CYCLE[i])
    ax.set_xlabel("environment steps")
    ax.set_ylabel("expected return")
    ax.set_title("Actor-critic with the n-step return: bias against variance")
    ax.set_ylim(-2.0, 1.1)
    ax.grid(alpha=0.3)
    ax.legend()
    np.save(HERE / "results" / "nstep_curves.npy",
            res_n[..., ::100].astype(np.float32))
    save_fig(fig, HERE / "figures" / "actor_critic_nstep.png")

    hard_case(rows)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / "actor_critic_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out/'actor_critic_summary.csv'}")


if __name__ == "__main__":
    main()
