"""REINFORCE with three baselines, on a discrete and a continuous problem.

Task
----
Implement REINFORCE and compare no baseline, a mean return baseline and the
variance minimising baseline, on a gridworld with a softmax policy and on
`Pendulum-v1` with a Gaussian policy.

Reference
---------
Sutton and Barto, 2nd edition, chapter 13.  The policy gradient theorem is
equation 13.5, REINFORCE is the box in section 13.3, baselines are section
13.4 and equation 13.11, the softmax in action preferences is section 13.2
and equation 13.9, and the Gaussian policy is section 13.7 and
equation 13.19.

The three baselines
-------------------
A baseline `b(s)` can be subtracted from the return without biasing the
gradient, because `E[ b(s) grad log pi(a|s) ] = b(s) grad sum_a pi(a|s) = 0`.
It changes only the variance.

`none`       b = 0.
`mean`       b = the mean return over the batch.  Cheap, and usually most of
             the available benefit.
`optimal`    the constant b that minimises the variance of the estimator,

                 b* = E[ ||grad log pi||^2 G ] / E[ ||grad log pi||^2 ]

             a gradient magnitude weighted average of the returns.  Derived by
             differentiating the variance with respect to b and setting it to
             zero; see Greensmith, Bartlett and Baxter, JMLR 2004, and the
             discussion around equation 13.11 in the book.

The form of `b*` has to match the form of the estimator it is used in.  The
estimator here is the per time step one, `sum_t (G_t - b) grad log pi_t`, so
the variance minimising constant is pooled over time steps,

    b* = sum_t E[ ||grad log pi_t||^2 G_t ] / sum_t E[ ||grad log pi_t||^2 ]

a gradient magnitude weighted average of the per step returns `G_t`.

The episode level form, with the total return `sum_t G_t` in the numerator,
looks equally defensible from the derivation, since the expectation there is
over trajectories.  It is not interchangeable.  On this gridworld
`sum_t G_t` is around -400 while `G_t` is around -8, so `b` lands an order of
magnitude away from the quantity it is subtracted from, every advantage
becomes a large positive number, the policy is pushed towards every action it
happened to take, and the expected return falls to -85.  Verified by running
it.

A note on the Gaussian parameterisation
---------------------------------------
`sigma` is the standard deviation everywhere: `np.random.normal` takes the
standard deviation, and the log gradient divides by the variance.  Passing
`sigma ** 2` to the sampler while still dividing by `sigma ** 2` in the
gradient leaves it off by a factor of `sigma ** 2`, which here would be 4 at
the start of training and 0.01 at the end, an uncontrolled learning rate
schedule hiding inside the annealing.  `test_dlog_gaussian_matches_a_
numerical_gradient` checks the gradient against finite differences.

Outputs
-------
figures/reinforce_gridworld.png
figures/reinforce_pendulum.png
results/reinforce_summary.csv
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
from rlkit.policies import (  # noqa: E402
    dlog_gaussian_probs, dlog_softmax_probs, gaussian_action, softmax_probs,
)

HERE = Path(__file__).resolve().parent
BASELINES = ["none", "mean", "optimal"]

GRIDWORLD = dict(
    env_id="Gym-Gridworlds/Penalty-3x3-v0",
    continuous=False,
    gamma=0.99,
    # At 0.1 the very first update saturates the softmax: with no baseline
    # the returns are all near -100, every action's preference is pushed down
    # by the same large amount, the policy becomes deterministic and
    # grad log pi goes to zero, so learning stops permanently after one
    # update.  0.01 leaves all three baselines able to learn, which is what
    # makes the comparison between them informative.  `step_size_robustness`
    # sweeps this deliberately.
    alpha=0.01,
    temperature=1.0,          # softmax temperature, held fixed
    episodes_per_update=10,
    max_steps=100_000,
    episodes_eval=1,
    n_centers=7,
    n_seeds=10,
    train_kwargs=dict(coordinate_observation=True, distance_reward=True,
                      max_episode_steps=50),
    eval_kwargs=dict(coordinate_observation=True, max_episode_steps=10),
)

PENDULUM = dict(
    env_id="Pendulum-v1",
    continuous=True,
    gamma=0.99,
    alpha=0.1,
    sigma0=2.0,
    sigma_min=0.1,
    episodes_per_update=10,
    max_steps=200_000,
    episodes_eval=10,
    n_centers=7,
    n_seeds=5,
    train_kwargs=dict(),
    eval_kwargs=dict(),
)


def build_features(env, n_centers):
    state_dim = env.observation_space.shape[0]
    low, high = env.observation_space.low, env.observation_space.high
    centers = make_centers(low, high, [n_centers] * state_dim, pad=0.1)
    sigmas = (np.asarray(high) - np.asarray(low)) / n_centers * 0.75 + 1e-8

    def get_phi(state):
        return rbf_features(np.asarray(state, float).reshape(-1, state_dim),
                            centers, sigmas)
    return get_phi, get_phi(env.reset()[0]).shape[1], state_dim


def collect_episodes(env, cfg, weights, get_phi, sigma, rng, seed_base):
    """Run `episodes_per_update` episodes, keeping each one separate.

    Episodes are kept separate rather than concatenated because both the
    return computation and the optimal baseline are per trajectory
    quantities.  Concatenating everything into one flat list and relying on
    a `done` flag works for the returns but silently merges trajectories
    wherever a per episode quantity is needed.
    """
    episodes = []
    for e in range(cfg["episodes_per_update"]):
        s, _ = reset_seeded(env, seed_base + e)
        phis, acts, rews = [], [], []
        done = False
        while not done:
            phi = get_phi(s)
            if cfg["continuous"]:
                a = gaussian_action(phi, weights, sigma, rng)
                step_a = np.clip(a, env.action_space.low, env.action_space.high)
            else:
                probs = softmax_probs(phi, weights, cfg["temperature"]).ravel()
                a = int(rng.choice(probs.shape[0], p=probs))
                step_a = a
            s, r, terminated, truncated, _ = env.step(step_a)
            phis.append(phi)
            acts.append(a)
            # Pendulum returns the reward as a 1 element array
            rews.append(float(np.asarray(r).ravel()[0]))
            done = terminated or truncated
        episodes.append((phis, acts, np.array(rews)))
    return episodes


def discounted_returns(rewards, gamma):
    """G_t for every t in one episode, computed backwards."""
    G = np.zeros(len(rewards))
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def reinforce(cfg, baseline, seed):
    """One REINFORCE run.  Returns the expected return logged per step."""
    env = make_env(cfg["env_id"], **cfg["train_kwargs"])
    env_eval = make_env(cfg["env_id"], **cfg["eval_kwargs"])
    get_phi, n_features, _ = build_features(env, cfg["n_centers"])
    rng = np.random.default_rng(seed)

    action_dim = (env.action_space.shape[0] if cfg["continuous"]
                  else env.action_space.n)
    weights = np.zeros((n_features, action_dim))
    sigma = cfg.get("sigma0", 1.0)

    history = np.zeros(cfg["max_steps"])
    tot_steps = 0
    seed_base = seed * 1_000_000

    while tot_steps < cfg["max_steps"]:
        episodes = collect_episodes(env, cfg, weights, get_phi, sigma, rng, seed_base)
        seed_base += cfg["episodes_per_update"]

        # per episode: returns, log gradients, and the pieces of the optimal
        # baseline.  Keeping them per episode is what makes b* correct.
        per_ep = []
        for phis, acts, rews in episodes:
            G = discounted_returns(rews, cfg["gamma"])
            grads = []
            for t, (phi, a) in enumerate(zip(phis, acts)):
                if cfg["continuous"]:
                    grads.append(dlog_gaussian_probs(phi, weights, sigma, a))
                else:
                    grads.append(dlog_softmax_probs(phi, weights, cfg["temperature"], a))
            per_ep.append((grads, G))

        if baseline == "mean":
            b = float(np.mean(np.concatenate([G for _, G in per_ep])))
        elif baseline == "optimal":
            # pooled over time steps, matching the per step estimator below
            num = den = 0.0
            for grads, G in per_ep:
                for g, Gt in zip(grads, G):
                    sq = float((g ** 2).sum())
                    num += sq * float(Gt)
                    den += sq
            b = num / den if den > 0 else 0.0
        else:
            b = 0.0

        # Accumulate the gradient: sum over time within an episode, mean
        # over episodes.  This is the REINFORCE estimator.  Dividing by the
        # total number of steps instead makes the effective learning rate
        # depend on how long the episodes happen to be, which stabilises
        # training for a reason that has nothing to do with the algorithm.
        grad = np.zeros_like(weights)
        n_steps = 0
        for grads, G in per_ep:
            for g, Gt in zip(grads, G):
                grad += (Gt - b) * g
            n_steps += len(G)
        weights = weights + cfg["alpha"] * grad / len(per_ep)

        if cfg["continuous"]:
            sigma = max(sigma - n_steps / cfg["max_steps"], cfg["sigma_min"])

        value = expected_return_linear(
            env_eval, weights, cfg["gamma"], get_phi, cfg["episodes_eval"],
            continuous=cfg["continuous"],
            action_low=env.action_space.low if cfg["continuous"] else None,
            action_high=env.action_space.high if cfg["continuous"] else None,
        )
        hi = min(tot_steps + n_steps, cfg["max_steps"])
        history[tot_steps:hi] = value
        tot_steps += n_steps

    env.close()
    env_eval.close()
    return history


def run(cfg, name, rows):
    (HERE / "results").mkdir(exist_ok=True)
    results = np.zeros((len(BASELINES), cfg["n_seeds"], cfg["max_steps"]))
    for i, baseline in enumerate(BASELINES):
        for seed in range(cfg["n_seeds"]):
            results[i, seed] = reinforce(cfg, baseline, seed)
        final = results[i, :, -1]
        # On the gridworld the per seed outcome is close to discrete: a seed
        # either finds the optimal route (0.941), wanders without reaching
        # the goal (0.0), or ends up on a penalty tile (about -9). A mean and
        # a standard deviation over ten such values are dominated by whichever
        # single seed went worst, so the fraction of seeds that solved the
        # task is reported alongside them as the more stable summary.
        solved = float(np.mean(final > 0.9 * final.max())) if final.max() > 0 else 0.0
        rows.append(dict(problem=name, baseline=baseline,
                         final_return=float(final.mean()),
                         final_return_median=float(np.median(final)),
                         final_return_std=float(final.std()),
                         fraction_solved=solved,
                         area_under_curve=float(results[i].mean())))
        print(f"    baseline {baseline:<8} final mean {final.mean():+8.3f} "
              f"median {np.median(final):+8.3f} +/- {final.std():6.3f}   "
              f"solved {solved:.0%}   mean over training {results[i].mean():+8.3f}")
    # stored at one point per 100 steps and in float32: the full array is
    # tens of megabytes and nothing downstream needs that resolution
    np.save(HERE / "results" / f"{name}_curves.npy",
            results[..., ::100].astype(np.float32))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, baseline in enumerate(BASELINES):
        error_shade_plot(ax, results[i], stepsize=1, smoothing_window=20,
                         label=baseline, color=CYCLE[i])
    ax.set_xlabel("environment steps")
    ax.set_ylabel("expected return")
    ax.set_title(f"REINFORCE on {cfg['env_id']}, {cfg['n_seeds']} seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, HERE / "figures" / f"reinforce_{name}.png")
    return results


def step_size_robustness(rows):
    """How large a step size each baseline survives.

    At a step size small enough for all three to work, every baseline reaches
    the optimum and the comparison says nothing.  The benefit of a baseline
    is variance reduction, and the visible consequence of lower variance is
    tolerance of a larger step.  This measures that directly.
    """
    print("  gridworld, step size robustness, 3 seeds")
    alphas = [0.003, 0.01, 0.03, 0.1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, baseline in enumerate(BASELINES):
        finals = []
        for alpha in alphas:
            cfg = dict(GRIDWORLD, alpha=alpha, n_seeds=3, max_steps=60_000)
            vals = [reinforce(cfg, baseline, seed)[-1] for seed in range(3)]
            finals.append(float(np.mean(vals)))
            rows.append(dict(problem=f"gridworld_alpha_{alpha}", baseline=baseline,
                             final_return=float(np.mean(vals)),
                             final_return_median=float(np.median(vals)),
                             final_return_std=float(np.std(vals)),
                             fraction_solved=float(np.mean(np.array(vals) > 0.9)),
                             area_under_curve=float("nan")))
        print(f"    {baseline:<8} " +
              "  ".join(f"a={a}: {v:+.3f}" for a, v in zip(alphas, finals)))
        ax.plot(alphas, finals, marker="o", label=baseline, color=CYCLE[i])
    ax.set_xscale("log")
    ax.set_xlabel("step size")
    ax.set_ylabel("expected return after 60k steps")
    ax.set_title("How large a step size each baseline tolerates")
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, HERE / "figures" / "reinforce_step_size.png")


def main() -> None:
    rows = []
    print("  gridworld, softmax policy")
    run(GRIDWORLD, "gridworld", rows)
    step_size_robustness(rows)
    print("  pendulum, Gaussian policy")
    run(PENDULUM, "pendulum", rows)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / "reinforce_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out/'reinforce_summary.csv'}")


if __name__ == "__main__":
    main()
