"""Policy evaluation by rollout, and reproducible seeding.

`expected_return` is the only honest performance measure in the tabular
chapters: the Bellman error says how self consistent the value estimates are,
not how much reward the greedy policy actually collects.  Assignment 5 asks
which of the three to use as the definition of performance, and the answer is
this one, so it is worth having a single trustworthy implementation.

On seeding
----------
Every learning routine in this repository takes an explicit
`numpy.random.Generator`, built from its `seed` argument at the top of the
run, and never touches the global `np.random` state.  Relying on the global
state makes a run depend on everything else that drew from it first, and the
standard library `random` module has a separate generator again that
`np.random.seed` does not affect.  Evaluation episodes are seeded
`0 .. episodes - 1` so that the same policy always scores the same number and
evaluation noise is not confounded with the learning curve.  See
`rlkit.envs.reset_seeded` for the part of seeding that is easy to miss.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .envs import reset_seeded

__all__ = ["expected_return", "expected_return_linear"]


def expected_return(
    env,
    Q: np.ndarray,
    gamma: float,
    episodes: int = 10,
) -> float:
    """Discounted return of the greedy policy w.r.t. a tabular Q, averaged.

    Episodes are seeded `0 .. episodes - 1` so that the same policy always
    scores the same number.  Without that, the evaluation noise is confounded
    with the learning curve.
    """
    total = np.zeros(episodes)
    for e in range(episodes):
        s, _ = reset_seeded(env, e)
        done, t = False, 0
        while not done:
            a = int(np.argmax(Q[s]))
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total[e] += (gamma ** t) * r
            t += 1
    return float(total.mean())


def expected_return_linear(
    env,
    weights: np.ndarray,
    gamma: float,
    get_phi: Callable[[np.ndarray], np.ndarray],
    episodes: int = 10,
    continuous: bool = False,
    action_low=None,
    action_high=None,
) -> float:
    """Same, for a linear approximator.

    With `continuous=False` the weights are read as Q(s, .) = phi(s) W and the
    greedy action is the argmax.  With `continuous=True` the weights are read
    as a deterministic mean action mu(s) = phi(s) W, clipped to the action
    range, which is how the Pendulum policy is evaluated.
    """
    total = np.zeros(episodes)
    for e in range(episodes):
        s, _ = reset_seeded(env, e)
        done, t = False, 0
        while not done:
            phi = get_phi(np.asarray(s, dtype=float))
            out = np.dot(phi, weights)
            if continuous:
                a = np.clip(out.ravel(), action_low, action_high)
            else:
                a = int(np.argmax(out.ravel()))
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total[e] += (gamma ** t) * r
            t += 1
    return float(total.mean())
