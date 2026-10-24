# 02. Dynamic programming: policy iteration, value iteration, GPI

**Book:** Sutton and Barto, 2nd edition, section 4.3 (policy iteration),
section 4.4 and equation 4.10 (value iteration), section 4.6 (generalized
policy iteration).
**Environment:** `Gym-Gridworlds/Penalty-3x3-v0`, `gamma = 0.99`,
`theta = 1e-10`, seven initial values `{-100, -10, -5, 0, 5, 10, 100}`.

```bash
python dynamic_programming.py
```

All three algorithms recover the optimal policy from all seven
initialisations, for both `v_*` and `q_*`. The script asserts it on every
run, so a regression fails loudly rather than producing a quiet wrong figure.

## Making the comparison mean something

The question is which algorithm converges in the fewest policy-evaluation
steps. That number only means something if all three count the same event,
so **one step is one Bellman backup applied to the whole state space**, and
exactly one Bellman error is logged per step, in every algorithm and for both
`V` and `Q`.

Two further choices follow from the same concern:

* All three use a **synchronous** sweep, computing the whole new value
  function from the whole old one. In-place Gauss-Seidel updates converge
  faster, so mixing the two styles would make the comparison partly a
  comparison of update style.
* GPI stops only when the policy is stable **and** the last evaluation sweep
  is below tolerance. Stopping on policy stability alone would let it finish
  with an unconverged value function, flattering its iteration count for a
  reason unrelated to the algorithm.

## Results

Policy evaluation steps to convergence, learning `v_*`:

| algorithm | -100 | -10 | -5 | 0 | 5 | 10 | 100 | mean ± std |
|---|---|---|---|---|---|---|---|---|
| VI | 8 | 8 | 8 | 8 | 168 | 237 | 466 | **129.0 ± 162.7** |
| PI | 3376 | 3465 | 3467 | 3470 | 3472 | 3474 | 3500 | **3460.6 ± 36.2** |
| GPI | 19 | 37 | 37 | 37 | 72 | 217 | 487 | **129.4 ± 158.9** |

Learning `q_*`:

| algorithm | -100 | -10 | -5 | 0 | 5 | 10 | 100 | mean ± std |
|---|---|---|---|---|---|---|---|---|
| VI | 9 | 9 | 9 | 9 | 169 | 238 | 467 | **130.0 ± 162.7** |
| PI | 3603 | 3691 | 3694 | 3696 | 3698 | 3700 | 3726 | **3686.9 ± 35.9** |
| GPI | 19 | 37 | 37 | 37 | 142 | 242 | 492 | **143.7 ± 160.7** |

### Which is fastest

VI and GPI are effectively tied at about 129 evaluation steps, and PI is
nearly thirty times slower. But the headline number hides the structure,
which is that **VI and GPI are fast for opposite reasons at opposite ends of
the initialisation range**:

* From pessimistic or neutral starts (`-100` to `0`), VI finishes in 8 steps
  and GPI needs 19 to 37. VI wins.
* From optimistic starts, GPI wins at `5` and `10` (72 and 217 against 168
  and 237), and VI wins at `100` (466 against 487).

Neither is uniformly better. The averages coincide only because the two
patterns happen to cancel across these seven values, which is a good
illustration of why a mean over initialisations is a weak summary.

### Why policy iteration is so much slower here

Not because policy improvement is expensive, but because PI insists on
evaluating the uniform random policy to convergence before it is allowed to
improve anything. Under that policy the induced Markov chain has cycles: the
agent wanders and can revisit states. Evaluating a cyclic chain converges
geometrically at rate `gamma = 0.99`, so reaching `1e-10` costs roughly
`ln(1/theta) / (1 - gamma)` sweeps, in the thousands.

Once the policy becomes deterministic and close to optimal, the induced chain
on this gridworld is acyclic with longest path 7, and evaluation terminates
*exactly* after finitely many sweeps rather than converging asymptotically.
GPI reaches that regime after its first improvement, five sweeps in. PI never
gets there cheaply because its first evaluation is the expensive one.

That makes the comparison tolerance-dependent, which is worth stating plainly:

| theta | VI | PI | GPI |
|---|---|---|---|
| 1e-4 | 129.0 | 1517.4 | 129.4 |
| 1e-6 | 129.0 | 2165.1 | 129.4 |
| 1e-8 | 129.0 | 2813.0 | 129.4 |
| 1e-10 | 129.0 | 3460.6 | 129.4 |

VI and GPI do not move at all, because they reach an exactly zero Bellman
error. PI grows by about 324 steps per decade of tolerance, without bound.
"PI is N times slower than VI" is therefore not a well-formed claim unless it
names `theta`.

### The Bellman error trend, and why it is not monotone

For GPI from `V0 = 0` the logged errors are

```
step:  1     2     3     4     5     6      7     8     9    10    11
err:  20.2  19.6  19.1  18.4  17.8  36.8   17.1  0.23  0.23  0.22  4.38
                                    spike                          spike
```

and increases occur at steps 6, 11, 16, 21, 26, 31 and 36: precisely every
fifth step. GPI improves the policy after every five evaluation sweeps, so
those are exactly the first sweeps following an improvement.

The mechanism is simple. Evaluation drives `V` towards `v_pi` for the current
`pi`, shrinking the residual of the *expectation* operator under that `pi`.
Improvement then replaces `pi` with a different, greedier policy, and `V` is
no longer close to the fixed point of the new operator, so the residual
jumps. A spike is evidence that policy improvement did something, not
evidence of a problem. The spikes shrink over successive rounds because each
new policy differs from the previous one in fewer states.

Value iteration has no improvement step and so none of those spikes, but its
curve is not monotone either. From `V0 = -100` the logged error *rises* for
three sweeps:

| sweep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| total absolute residual (L1) | 125.0 | 205.9 | **290.0** | 187.3 | 97.2 | 9.78 | 9.60 | 0 |
| largest residual (max norm) | 101.0 | 99.99 | 98.99 | 98.00 | 97.02 | 9.69 | 9.60 | 0 |

There is no contradiction. The Bellman optimality operator is a contraction
of modulus `gamma` in the **max norm**, not in the L1 norm being logged, and
the second row shows that contraction holding exactly: `101 * 0.99 = 99.99`,
`99.99 * 0.99 = 98.99`, and so on. Meanwhile corrections are propagating
outward from the states with rewards, so the *number* of states carrying a
nonzero residual grows faster than each residual shrinks, and the sum rises.
The L1 Bellman error is a perfectly good convergence diagnostic, but it is
not the quantity the contraction theorem is about.

### Is this gridworld a good showcase for GPI?

Yes, and the reason is the mechanism above. GPI pays off when evaluating a
policy to convergence is far more expensive than finding a better policy, and
that gap is extreme here: the random starting policy is the single most
expensive policy in the problem to evaluate, and GPI abandons it after five
sweeps. GPI would show less benefit where the greedy operator changes the
policy slowly, for instance from a near-deterministic start already close to
optimal, or in a problem where every policy induces an acyclic chain so that
evaluation is cheap for all of them.

### Learning `v_*` versus `q_*`

Almost identical. `Q` costs one extra evaluation step for VI across the board
(9 against 8) and about 6 percent more for PI, which is the cost of
propagating through `|A| = 5` action values rather than one state value, and
GPI matches except at the two mid-optimistic starts. No algorithm changes
rank. The `Q` representation uses five times the memory and buys nothing in
the model-based setting; its advantage is that it needs no model at action
selection time, and that matters from chapter 03 onward, not here.

### Does initialisation matter, and is one value always worse?

Yes, and `100` is always the worst, for every algorithm and both value
functions. Optimistic values have to be pushed back down, and at
`gamma = 0.99` each backup removes about one percent of the excess. The
pessimistic end is not symmetric: `-100` is the *cheapest* start for VI and
GPI, because the optimality backup takes a max, so a uniformly too-low value
function is corrected in one sweep wherever a reward is available, and the
acyclic exact convergence then takes over. The value range of this problem is
`[-9.02, 1.0]`, so `-100` is further from the truth than `100` in absolute
terms and still converges faster. Distance from the solution is not what
drives the cost; the direction of the error relative to the max operator is.

PI is the least sensitive to initialisation, with a standard deviation of 36
against 160 for the other two, but only because its cost is dominated by the
fixed expense of evaluating the random initial policy, which swamps the
initialisation effect. That is low variance for an unflattering reason.

## Figures

| file | content |
|---|---|
| `figures/bellman_error_V.png` | 3 algorithms × 7 initialisations, learning `v_*` |
| `figures/bellman_error_Q.png` | the same, learning `q_*` |
| `results/iteration_counts.csv` | every count |
| `results/summary_table.md` | the tables above, regenerated on each run |
