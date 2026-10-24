# The environments, and why the version is pinned

Every tabular chapter uses `Gym-Gridworlds` by Simone Parisi,
<https://github.com/sparisi/gym_gridworlds>. Chapter 07 also uses
`Pendulum-v1` from `gymnasium`.

## Penalty-3x3-v0

Nine states, five actions, deterministic.

```
 s0      s1(-10)  s2(+1, goal)
 s3      s4(-10)  s5
 s6      s7       s8
```

The agent starts at `s0`. Stepping onto a red tile costs `-10`. The `+1` is
paid only for choosing `STAY` on the green tile, and that transition ends the
episode. It is the only terminal transition in the MDP.

With `gamma = 0.99` the optimal policy is

```
DOWN   RIGHT  STAY
DOWN   RIGHT  UP
RIGHT  RIGHT  UP
```

and the optimal values are

```
 0.941  -9.01   1.00
 0.951  -9.02   0.99
 0.961   0.97   0.98
```

The route `0 -> 3 -> 6 -> 7 -> 8 -> 5 -> 2` is six moves, so
`v_*(s0) = 0.99^6 = 0.9415`. Several chapters assert this, and
`tests/test_rlkit.py` checks it to within `1e-12`.

## Empty-2x2-v0 (chapter 06)

Four states in a 2 by 2 grid, used with `coordinate_observation=True`,
`random_action_prob=0.1` and `reward_noise_std=0.01`. Small and stochastic,
which is what the fitted Q-iteration exercise needs: the point of the dataset
size hyperparameter is averaging out noise.

## DangerMaze-6x6-v0 (chapters 05 and 08)

Thirty states on a 5 by 6 grid with pits worth `-100`, two tiles worth `-10`
and a goal worth `+1`. Its optimal value function has real discontinuities,
from `-99` next to `+0.95`, which is why chapter 05 uses it to compare
feature constructions: an empty grid gives a smooth surface that every
construction fits almost perfectly and so cannot tell them apart.

## Why the pin matters

See `requirements.txt`. In short, the action encoding changed after this
course ran, and two of the three environment ids above were removed. Neither
change raises an error; they just make the code mean something different.
`rlkit/envs.py` resolves action names and environment ids at import time so
that the code keeps working either way, and `requirements.txt` pins the
October 2024 commit so the published numbers reproduce exactly.
