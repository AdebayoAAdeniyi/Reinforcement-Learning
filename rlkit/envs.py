"""Environment helpers: tabular MDP extraction and version robust action names.

Everything in this repository is built on `gym_gridworlds` by Simone Parisi,
https://github.com/sparisi/gym_gridworlds.  That package is a moving target.
Between the October 2024 state used by the CMPUT 655 assignments and today the
upstream repository changed two things that break code silently:

1.  The action encoding.  In October 2024 it was
    LEFT 0, DOWN 1, RIGHT 2, UP 3, STAY 4.  Later it became
    LEFT 0, RIGHT 1, DOWN 2, UP 3, STAY 4.  Any policy written as integers
    therefore means something different depending on which commit is installed,
    and nothing raises an error.
2.  Two environment ids were removed.  `Empty-2x2-v0` became
    `Empty-RandomStart-2x2-v0` and `DangerMaze-6x6-v0` became
    `DangerMaze-5x6-v0`.

`requirements.txt` pins the October 2024 commit so the numbers in this
repository reproduce exactly.  On top of that pin, this module never writes a
bare action integer.  Actions are referred to by name and resolved at import
time from whatever version happens to be installed, and environment ids are
resolved against the registry.  The code therefore keeps working, and keeps
meaning the same thing, if the pin is ever lifted.
"""

from __future__ import annotations

import numpy as np

import gymnasium
import gym_gridworlds  # noqa: F401  (registers the Gym-Gridworlds/* ids)
from gym_gridworlds import gridworld as _gw

__all__ = [
    "ACTIONS",
    "ACTION_NAMES",
    "action_index",
    "policy_from_names",
    "names_from_policy",
    "resolve_env_id",
    "make_env",
    "reset_seeded",
    "extract_mdp",
    "MDP",
]

#: Action name -> integer index, read from the installed package rather than
#: hardcoded, so this stays correct across upstream reorderings.
ACTIONS: dict[str, int] = {
    name: int(getattr(_gw, name)) for name in ("LEFT", "DOWN", "RIGHT", "UP", "STAY")
}

#: Integer index -> action name, in index order.
ACTION_NAMES: list[str] = [name for name, _ in sorted(ACTIONS.items(), key=lambda kv: kv[1])]


def action_index(name: str) -> int:
    """Return the integer index of a named action."""
    try:
        return ACTIONS[name.upper()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown action {name!r}, expected one of {sorted(ACTIONS)}") from exc


def policy_from_names(grid: list[list[str]]) -> np.ndarray:
    """Turn a grid of action names into a flat array of action indices.

    >>> policy_from_names([["DOWN", "RIGHT", "STAY"]])
    array([1, 2, 4])

    Writing the optimal policy this way is the whole point: the array above is
    correct under the October 2024 encoding and would be wrong under the
    current one, but the call is correct under both.
    """
    return np.array([action_index(a) for row in grid for a in row], dtype=int)


def names_from_policy(policy: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Inverse of :func:`policy_from_names`, for printing."""
    names = np.array(ACTION_NAMES)[np.asarray(policy, dtype=int)]
    return names.reshape(shape) if shape is not None else names


#: Environment ids that were renamed upstream, mapped to their replacements.
_ID_ALIASES = {
    "Gym-Gridworlds/Empty-2x2-v0": "Gym-Gridworlds/Empty-RandomStart-2x2-v0",
    "Gym-Gridworlds/DangerMaze-6x6-v0": "Gym-Gridworlds/DangerMaze-5x6-v0",
}


def resolve_env_id(env_id: str) -> str:
    """Return `env_id` if it is registered, otherwise its known replacement.

    Raises if neither is available, instead of letting a `gymnasium` error
    surface with no hint that the cause is a package version mismatch.
    """
    if env_id in gymnasium.registry:
        return env_id
    alt = _ID_ALIASES.get(env_id)
    if alt is not None and alt in gymnasium.registry:
        return alt
    raise gymnasium.error.NameNotFound(
        f"{env_id!r} is not registered by the installed gym_gridworlds, and no "
        f"known replacement is either. Install the pinned commit listed in "
        f"requirements.txt."
    )


def make_env(env_id: str, **kwargs):
    """`gymnasium.make` with environment id resolution."""
    return gymnasium.make(resolve_env_id(env_id), **kwargs)


def reset_seeded(env, seed: int):
    """Reset an environment so the whole episode stream is reproducible.

    `env.reset(seed=...)` seeds the environment's own generator but **not**
    the generator belonging to `env.action_space`.  Those are separate, and
    the space's generator is initialised from operating system entropy when
    the space is built.

    That matters here because `gym_gridworlds` implements its
    `random_action_prob` by calling `self.action_space.sample()`:

        if self.np_random.random() < self.random_action_prob:
            action = self.action_space.sample()

    So with `random_action_prob=0.1`, one action in ten is drawn from an
    unseeded generator, and a run that looks fully seeded is not reproducible
    at all.  Nothing in any figure reveals this, which is why there is a test
    for it; see docs/ENGINEERING_NOTES.md section 3.

    Always use this instead of `env.reset(seed=...)` at the start of a run.
    """
    out = env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    env.observation_space.seed(int(seed))
    return out


class MDP:
    """The three tables that define a finite MDP with deterministic transitions.

    Attributes
    ----------
    R : (n_states, n_actions)
        Expected reward r(s, a).
    P : (n_states, n_actions, n_states)
        Transition probabilities p(s' | s, a), left exactly as the environment
        reports them.  Termination is *not* folded into P here.  It is applied
        explicitly as a `(1 - T[s, a])` factor inside every Bellman operator in
        `rlkit.bellman`, so that the correct indexing is visible at the point
        where it matters rather than hidden in a preprocessing step.
    T : (n_states, n_actions)
        1.0 if taking action a in state s ends the episode, else 0.0.
        Note the indexing: termination is a property of the transition that
        *leaves* (s, a).  `T[s_next, a]` is a different quantity, nonzero in
        states the agent merely passes through, and it moves the fixed point
        of the operator; see docs/ENGINEERING_NOTES.md section 2.
    """

    __slots__ = ("R", "P", "T", "n_states", "n_actions")

    def __init__(self, R: np.ndarray, P: np.ndarray, T: np.ndarray):
        self.R, self.P, self.T = R, P, T
        self.n_states, self.n_actions = R.shape

    def __iter__(self):
        """Allow `R, P, T = mdp`."""
        return iter((self.R, self.P, self.T))


def extract_mdp(env_id: str = "Gym-Gridworlds/Penalty-3x3-v0", **kwargs) -> MDP:
    """Read R, P and T out of a tabular gridworld by brute force enumeration.

    This is the snippet the assignments hand out, with one difference.  It
    calls `env.unwrapped.set_state`, because `gymnasium.make` returns a
    `TimeLimit`/`OrderEnforcing` wrapper that does not expose `set_state`, so
    the snippet as written fails on current gymnasium.

    The environment is created without reward noise on purpose.  R must be the
    *expected* reward, and a single noisy sample per (s, a) is not that.
    """
    env = make_env(env_id, **kwargs)
    n_states, n_actions = env.observation_space.n, env.action_space.n

    R = np.zeros((n_states, n_actions))
    P = np.zeros((n_states, n_actions, n_states))
    T = np.zeros((n_states, n_actions))

    env.reset()
    for s in range(n_states):
        for a in range(n_actions):
            env.unwrapped.set_state(s)
            s_next, r, terminated, _, _ = env.step(a)
            R[s, a] = r
            P[s, a, s_next] = 1.0
            T[s, a] = float(terminated)
    env.close()
    return MDP(R, P, T)
