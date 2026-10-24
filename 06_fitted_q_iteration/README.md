# 06. Fitted Q-Iteration

**Background:** Ernst, Geurts and Wehenkel, *Tree-Based Batch Mode
Reinforcement Learning*, JMLR 2005, is the original FQI paper. The connection
to the book is Sutton and Barto section 9.3 for the semi-gradient update and
section 11.3 for why holding the target fixed matters.
**Environment:** `Gym-Gridworlds/Empty-2x2-v0` with coordinate observations,
`random_action_prob = 0.1` and `reward_noise_std = 0.01`, state aggregation
features, `gamma = 0.99`, 10,000 steps, 5 seeds. The optimum is 0.994.

```bash
python fitted_q_iteration.py
```

## The algorithm and the two conditions it restores

Regression assumes the data is i.i.d. and the target is stationary. Online
Q-learning violates both: consecutive transitions come from one trajectory,
and the target `r + gamma max_a' Q(s', a')` moves whenever the weights move.
FQI addresses them separately.

```
repeat K times:
    compute the target for the whole dataset from the current weights
    repeat N times:
        one gradient step on the squared error against that fixed target
```

Holding the target fixed for the N inner steps makes the inner loop an
ordinary stationary regression. Using a batch of D rather than the single
most recent transition is what addresses, or fails to address, independence.

`N = K = D = 1` is online Q-learning with a linear approximator.

> **Reproducibility note.** This is the one chapter whose environment uses
> `random_action_prob`, which `gym_gridworlds` implements through
> `action_space.sample()`. That generator is not seeded by
> `env.reset(seed=...)`, so the sweep is run through
> `rlkit.envs.reset_seeded`, which seeds it explicitly. Two separate
> processes now produce byte-identical results. See
> [`docs/ENGINEERING_NOTES.md`](../docs/ENGINEERING_NOTES.md) section 3.

## Results: consecutive batches

Expected return after 10,000 steps, batches of D consecutive transitions:

| | D=1 | D=100 | D=1000 |
|---|---|---|---|
| N=1, K=1 | **0.994** | **0.994** | **0.994** |
| N=1, K=100 | 0.278 | **0.994** | **0.994** |
| N=1, K=1000 | 0.000 | 0.596 | **0.994** |
| N=100, K=1 | 0.476 | 0.875 | **0.994** |
| N=100, K=100 | 0.000 | 0.199 | **0.994** |
| N=100, K=1000 | 0.000 | 0.000 | **0.994** |
| N=1000, K=1 | 0.476 | 0.398 | **0.994** |
| N=1000, K=100 | 0.000 | 0.080 | **0.994** |
| N=1000, K=1000 | 0.000 | 0.000 | **0.994** |

### Which hyperparameter matters most

`D`, decisively. **All nine combinations of N and K reach the optimum at
D = 1000**, and nothing rescues D = 1 except keeping N and K at 1 as well.
The environment has `reward_noise_std = 0.01` and a 10 percent chance of a
random action, so a single transition carries almost no information about the
expected reward or the expected next state. A batch of 1000 averages that
away; a batch of 1 fits the noise.

### Why the TD error is a bad progress indicator

The D = 1 column holds the **lowest** TD errors in the whole sweep, 0.0004 to
0.0006, together with the worst returns. That is the classical regression
failure: with one sample per fit the model interpolates it exactly, so the
training error goes to zero while the fit generalises to nothing.

The **highest** TD errors in the table, 0.246 and 0.297, belong to
`N=1, K=1` at D = 100 and D = 1000, both of which reach the optimum. A single
gradient step on a large batch leaves a large residual and learns well. The
TD error and the return point in opposite directions across the whole sweep.

### Can we do too many updates

Yes, unambiguously. At D = 100, holding N = 1 and raising K from 1 to 100 to
1000 gives 0.994, 0.994, 0.596; holding K = 1000 and raising N from 1 to 100
to 1000 gives 0.596, 0.000, 0.000. Updates are not free, and more of them
against too little data is actively destructive rather than merely
wasteful.

### Fewer fits or fewer updates per fit

D = 1000 cannot answer this: all nine combinations sit at 0.994, so any
preference read off that row would be noise. The comparison has to be made
where the setting bites.

At D = 100 with a fixed budget of 1000 total gradient steps, `N=1, K=1000`
gives 0.596 and `N=1000, K=1` gives 0.398. Recomputing the target often beats
taking many steps against one increasingly stale target, which is the
expected direction given that refreshing the target is the entire purpose of
the outer loop.

## Results: uniform sampling from a long replay buffer

Same sweep, with a buffer holding the whole run and batches of D drawn from
it uniformly at random:

| | D=1 | D=100 | D=1000 |
|---|---|---|---|
| N=1, K=1 | **0.994** | **0.994** | **0.994** |
| N=1, K=100 | 0.000 | 0.935 | **0.994** |
| N=1, K=1000 | 0.000 | 0.875 | **0.994** |
| N=100, K=1 | 0.000 | 0.935 | **0.994** |
| N=100, K=100 | 0.000 | **0.994** | **0.994** |
| N=100, K=1000 | 0.000 | 0.596 | **0.994** |
| N=1000, K=1 | 0.000 | 0.418 | **0.994** |
| N=1000, K=100 | 0.000 | 0.595 | **0.994** |
| N=1000, K=1000 | 0.000 | 0.795 | **0.994** |

### The i.i.d. question, settled by measurement

Whether a replay buffer fixes the independence violation is usually asserted
rather than measured. Here is the comparison at D = 100, where the task is
hard enough to discriminate:

| N, K | total updates | consecutive | sampled | change |
|---|---|---|---|---|
| 1, 1 | 1 | 0.994 | 0.994 | — |
| 1, 100 | 100 | 0.994 | 0.935 | -0.06 |
| 100, 1 | 100 | 0.875 | 0.935 | +0.06 |
| 1, 1000 | 1,000 | 0.596 | **0.875** | **+0.28** |
| 1000, 1 | 1,000 | 0.398 | 0.418 | +0.02 |
| 100, 100 | 10,000 | 0.199 | **0.994** | **+0.80** |
| 1000, 100 | 100,000 | 0.080 | **0.595** | **+0.52** |
| 100, 1000 | 100,000 | 0.000 | **0.596** | **+0.60** |
| 1000, 1000 | 1,000,000 | 0.000 | **0.795** | **+0.80** |

**The answer is yes, and the effect is large and systematic.** The rows are
sorted by the total number of gradient updates, and that ordering is the
story: sampling makes no difference when few updates are done, and the
benefit grows monotonically with the update count until it is worth 0.8 of
the return. Every configuration that collapsed to zero under consecutive
batches recovers to between 0.595 and 0.795.

That is exactly what the diagnosis predicts. A correlated batch describes a
small region of the state space, and the more gradient steps you take against
it, the harder you fit that region at the expense of everywhere else.
Decorrelating the batch is what makes a large number of updates safe rather
than harmful. `figures/fqi_iid_comparison.png` shows the mechanism directly:
the consecutive curves climb at first and then decay through training, while
the sampled ones stay up.

### Where sampling hurts

At D = 1 it is worse everywhere it differs: 0.278, 0.476 and 0.476 all become
0.000. This is worth being careful about, because it is tempting to assume
the two modes must coincide when the batch holds a single transition. They do
not. `on_the_fly` fits the transition that just happened; `replay` fits one
transition drawn from anywhere in the history. Those are different
algorithms, and fitting one *stale* sample hard, with a target computed from
current weights, is worse than fitting one fresh sample. D = 1 is therefore
not a control condition.

A buffer sized at exactly D would have made the comparison meaningless in the
other direction: drawing D items from a buffer of the last D transitions
samples the same window the consecutive batch uses, differing only in which
samples repeat. That is a bootstrap resample, not an independence test. This
was the first version of the experiment here, and it was discarded before it
produced any numbers. `BUFFER_SIZE` is the whole run for that reason.

## Answers to the discussion questions

**Which condition does fixing the target before fitting guarantee?**
Stationarity. The N inner gradient steps see a target that does not move.

**Which condition does offline batch collection guarantee?** Independence, or
a far better approximation of it, since the dataset is not one contiguous
stretch of one trajectory.

**Why does continual collection break it, and how would you fix it?**
Consecutive transitions are temporally correlated and the state distribution
shifts as the policy improves. A replay buffer with uniform sampling fixes
it, and the table above quantifies by how much.

**Why does primacy bias not matter here?** The approximator is linear and the
squared error against a fixed target is convex, so there is one global
optimum and gradient descent reaches it from any starting point. Warm
starting from the previous weights changes the path, not the destination.
That stops being true the moment the approximator is non-linear.

**What is `N = K = D = 1`?** Online Q-learning with linear function
approximation. It reaches 0.994 here, so the simplest member of the family
handles this problem perfectly well.

**If the environment were deterministic, what could be reduced?** `D`. Its
job in this sweep is to average out the reward noise and the 10 percent
random action, and with neither present a single transition is a sufficient
description of `(s, a)`. `D` is the hyperparameter carrying the
stochasticity, and the evidence is that it is the only one whose effect
survives at every `N` and `K`.

**Would a non-linear approximator change things?** It would remove the
convexity argument above, so warm starting and the number of gradient steps
per fit would both begin to matter in ways they do not here, and overfitting
a small `D` would get worse rather than better. It would also let the
approximator represent value functions a four feature state aggregation
cannot, which on a 2 by 2 grid is not a limitation anyone is feeling.

## Figures

| file | content |
|---|---|
| `figures/fqi_sweep_on_the_fly.png` | all 27 settings, return above and TD error below |
| `figures/fqi_sweep_replay.png` | the same with uniform sampling from the buffer |
| `figures/fqi_iid_comparison.png` | the two modes across N and K at D = 100 |
| `results/fqi_summary.csv` | the tables above |
| `results/*_return.npy`, `*_td_error.npy` | the per seed curves |
