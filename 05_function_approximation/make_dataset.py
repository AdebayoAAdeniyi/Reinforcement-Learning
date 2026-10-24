"""Generate the gridworld transition dataset used by `semi_gradient_td.py`.

The course-supplied dataset for this exercise is not redistributable, so
rather than ship a file whose provenance cannot be stated, this script
builds an equivalent one from an environment pinned in `requirements.txt`.
The chapter is therefore reproducible from scratch with one command, and the
dataset's construction is visible and adjustable rather than opaque.

What is generated
-----------------
`Gym-Gridworlds/DangerMaze-6x6-v0`, a 5 by 6 grid with pits worth -100, two
tiles worth -10 and a goal worth +1.  It is chosen over a plain empty grid
because its optimal value function has genuine discontinuities: a tile worth
-99 can sit next to one worth +0.95.  A smooth gradient of the kind an empty
grid produces is fitted almost perfectly by every one of the five feature
constructions, so it cannot distinguish between them.

Transitions are collected under an epsilon-greedy version of the optimal
policy with epsilon = 0.2, starting from every state in turn so that the
whole grid including the pits is covered.  The behaviour policy is saved with
the data, because the value function the chapter estimates is `v_pi` of that
policy and the comparison would be meaningless without it.

    python make_dataset.py        writes data/dangermaze.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit.bellman import optimal_q_backup, solve_q_exactly, solve_v_exactly  # noqa: E402
from rlkit.envs import extract_mdp, make_env, reset_seeded  # noqa: E402
from rlkit.policies import eps_greedy_probs_all  # noqa: E402

HERE = Path(__file__).resolve().parent
ENV_ID = "Gym-Gridworlds/DangerMaze-6x6-v0"
GRID = (5, 6)
GAMMA = 0.99
BEHAVIOUR_EPS = 0.05
EPISODES_PER_START = 20
MAX_EPISODE_STEPS = 30
SEED = 0


def build():
    mdp = extract_mdp(ENV_ID)

    # optimal Q by value iteration on the true model, used as the reference
    Q_star = np.zeros((mdp.n_states, mdp.n_actions))
    for _ in range(5000):
        Q_star = optimal_q_backup(Q_star, mdp, GAMMA)

    # the behaviour policy and the value function the chapter will estimate
    pi = eps_greedy_probs_all(Q_star, BEHAVIOUR_EPS)
    V_pi = solve_v_exactly(pi, mdp, GAMMA)
    Q_pi = solve_q_exactly(pi, mdp, GAMMA)

    env = make_env(ENV_ID, coordinate_observation=True,
                   max_episode_steps=MAX_EPISODE_STEPS)
    rng = np.random.default_rng(SEED)
    S, A, R, S_next, TERM, S_idx, S_next_idx = [], [], [], [], [], [], []

    for start in range(mdp.n_states):
        for ep in range(EPISODES_PER_START):
            reset_seeded(env, SEED * 10_000 + start * 100 + ep)
            # with coordinate_observation the environment stores the agent
            # position as a (row, col) tuple, so set_state takes coordinates
            # rather than the flat index it takes in the tabular case
            env.unwrapped.set_state(tuple(int(v) for v in np.unravel_index(start, GRID)))
            s_idx = start
            for _ in range(MAX_EPISODE_STEPS):
                s_xy = np.array(np.unravel_index(s_idx, GRID), dtype=float)
                a = int(rng.choice(mdp.n_actions, p=pi[s_idx]))
                obs, r, terminated, truncated, _ = env.step(a)
                s_next_idx = int(np.ravel_multi_index(
                    np.round(np.asarray(obs)).astype(int), GRID))
                S.append(s_xy)
                A.append(a)
                R.append(r)
                S_next.append(np.asarray(obs, dtype=float))
                TERM.append(float(terminated))
                S_idx.append(s_idx)
                S_next_idx.append(s_next_idx)
                if terminated or truncated:
                    break
                s_idx = s_next_idx
    env.close()

    out = HERE / "data"
    out.mkdir(exist_ok=True)
    path = out / "dangermaze.npz"
    np.savez_compressed(
        path,
        s=np.array(S), a=np.array(A), r=np.array(R),
        s_next=np.array(S_next), term=np.array(TERM),
        s_idx=np.array(S_idx), s_next_idx=np.array(S_next_idx),
        pi=pi, V_pi=V_pi, Q_pi=Q_pi, Q_star=Q_star,
        grid=np.array(GRID), gamma=GAMMA, behaviour_eps=BEHAVIOUR_EPS,
    )
    print(f"  {len(S)} transitions from {mdp.n_states} start states")
    print(f"  states visited: {len(set(S_idx))} of {mdp.n_states}")
    print(f"  v_pi range [{V_pi.min():.2f}, {V_pi.max():.2f}]")
    print(f"  wrote {path.relative_to(HERE)}")
    return path


if __name__ == "__main__":
    build()
