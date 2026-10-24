# 01. Bellman expectation equations

Computing `v_pi` and `q_pi` of a known policy by iterative policy evaluation,
and reading the Bellman error as it converges.

**Book:** Sutton and Barto, 2nd edition, equations 3.12 to 3.14 (the Bellman
expectation equations) and section 4.1 (iterative policy evaluation).
**Environment:** `Gym-Gridworlds/Penalty-3x3-v0`.

```bash
python bellman_equations.py
```

## The problem

`Penalty-3x3-v0` is a nine-state deterministic gridworld. The agent starts at
the top left (state 0) and must reach the green tile (state 2). The reward is
`+1`, given only for choosing `STAY` on the green tile, and that transition
ends the episode. Walking onto either red tile (states 1 and 4) costs `-10`.

The optimal policy is written out by hand and then verified against value
iteration:

```
DOWN   RIGHT  STAY
DOWN   RIGHT  UP
RIGHT  RIGHT  UP
```

It routes the agent `0 -> 3 -> 6 -> 7 -> 8 -> 5 -> 2`, six moves around both
red tiles, so `v_pi(0)` should be exactly `gamma^6`. At `gamma = 0.99` that is
`0.99^6 = 0.9415`, and the script prints `0.9415`. A closed-form prediction
that the code has to match is worth more than a plot that merely looks
plausible, and the test suite checks it to within `1e-12`.

## Two implementation details that decide whether the answer is right

### Policies are written as names, not integers

The obvious way to write that policy is `[[1, 2, 4], [1, 2, 3], [2, 2, 3]]`.
Those integers belong to the gridworld package, which reordered its actions
between releases, so the same array means a different policy depending on
which version is installed and nothing raises an error.
`rlkit.envs.policy_from_names` resolves the names against whatever is
installed, so the policy keeps its meaning either way. See
[`docs/ENGINEERING_NOTES.md`](../docs/ENGINEERING_NOTES.md) section 1.

### Termination belongs to the transition

```
v(s) = sum_a pi(a|s) [ R[s,a] + (1 - T[s,a]) * gamma * sum_s' P[s,a,s'] v(s') ]
```

`T[s, a]` is 1 when the transition *leaving* `(s, a)` ends the episode. Here
the only such transition is `(state 2, STAY)`. `T[s_next, a]` asks a
different question, is nonzero in states the agent merely passes through, and
moves the fixed point without making any figure look wrong.

The script asserts `v_pi(s) = sum_a pi(a|s) q_pi(s,a)` after every run. Two
Bellman operators that disagree on that identity cannot both be right, and
asserting it at the point of use costs nothing.

## Results

Total absolute Bellman error `sum_s |T v - v|` after each sweep, for `v_pi`:

| sweep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `gamma=0.99`, `V0=0`   | 21.0 | 1.98 | 1.96 | 0.97 | 0.961 | 0.951 | 0.941 | **0** |
| `gamma=0.99`, `V0=-10` | 31.4 | 22.4 | 22.0 | 11.0 | 10.8 | 10.6 | 10.4 | **0** |
| `gamma=0.99`, `V0=10`  | 29.8 | 18.4 | 18.0 | 9.02 | 8.84 | 8.65 | 8.47 | **0** |
| `gamma=0.01`, `V0=0`   | 21.0 | 0.02 | 2e-4 | 1e-6 | 1e-8 | 1e-10 | <1e-10 | |

### How initialisation affects convergence

In this environment it changes the size of the Bellman error but not the
number of sweeps. Every `gamma = 0.99` run finishes in exactly 8 sweeps
whatever the initial value, and the reason is specific and checkable: the
chain induced by a deterministic policy on this gridworld is acyclic, with
longest path 7 from state 0 to the terminal transition. Once correct values
have propagated backwards along that path, one further sweep returns an error
of exactly zero rather than merely a small one. The usual contraction
argument, which predicts the error shrinking by a factor `gamma` per sweep
forever, describes the intermediate sweeps but is not what ends the run.

Initialisation still sets the error magnitude. Starting at 0 is closest to
the true values, which lie in `[-9.02, 1.0]`, so its curve begins an order of
magnitude below the others and stays there.

### How gamma affects the final values and the speed

`gamma` changes the answer completely. At `gamma = 0.01` the future is worth
almost nothing, so `v_pi(0) = 0.01^6 = 1e-12`, numerically zero, and the
value function is flat everywhere except on tiles that pay immediately. At
`gamma = 0.99` the `+1` six steps away is still worth `0.94`, and the value
function has a clear gradient along the optimal route.

The effect on speed runs against the usual intuition. `gamma = 0.01` decays
geometrically with ratio `0.01` and reaches the `1e-10` tolerance in 7
sweeps; `gamma = 0.99` reaches *exact* zero in 8. The determinism of the
gridworld, not the discount factor, sets the iteration count here.

### Would gamma matter more in a larger gridworld?

Yes, and the mechanism above says why. The acyclic shortcut that ends the run
in 8 sweeps depends on the longest path through the induced chain, which
grows with grid size. In a larger grid the contraction rate has room to
become the binding constraint, and at that point a `gamma` near 1 costs
roughly `1 / (1 - gamma)` sweeps to reach a given tolerance: a hundred times
more work at `0.99` than at `0.5`. It also matters more because a larger grid
puts the goal further away, so the discounted signal reaching the start
state, `gamma^d`, becomes small enough that a small `gamma` stops
distinguishing good states from bad ones at all.

## Figures

| file | content |
|---|---|
| `figures/v_pi_init_{-10,0,10}.png` | heatmap of `v_pi` and the Bellman error trend, one column per gamma |
| `figures/q_pi_init_{-10,0,10}.png` | one heatmap per action plus the error trend |
| `results/bellman_errors.csv` | every logged Bellman error in long format |
| `results/value_functions.csv` | the converged `v_pi` and `q_pi` |
