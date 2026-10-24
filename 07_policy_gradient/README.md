# 07. Policy gradient: REINFORCE and its baselines

**Book:** Sutton and Barto, 2nd edition. The policy gradient theorem is
equation 13.5, REINFORCE is the box in section 13.3, baselines are section
13.4 and equation 13.11, soft-max in action preferences is section 13.2 and
equation 13.9, the Gaussian policy is section 13.7 and equation 13.19.
**Environments:** `Gym-Gridworlds/Penalty-3x3-v0` with a softmax policy over
49 RBF features, and `Pendulum-v1` with a Gaussian policy over 343 RBF
features.

```bash
python reinforce.py
```

## The three baselines

A baseline `b` can be subtracted from the return without biasing the
gradient, because `E[b * grad log pi(a|s)] = b * grad sum_a pi(a|s) = 0`. It
changes only the variance.

* `none`: `b = 0`.
* `mean`: the mean return over the batch. Cheap.
* `optimal`: the constant that minimises the variance of the estimator,
  `b* = sum_t E[||grad log pi_t||^2 G_t] / sum_t E[||grad log pi_t||^2]`,
  a gradient magnitude weighted average of the per step returns.

`tests/test_rlkit.py::test_dlog_softmax_sums_to_zero_over_actions` checks the
identity that makes a baseline unbiased, and the log gradients are checked
against numerical differentiation.

### The optimal baseline has to match the estimator

The form of `b*` must match the form of the estimator it is used in. The
estimator here is the per time step one, `sum_t (G_t - b) grad log pi_t`, so
the variance minimising constant is pooled over time steps, as written above.

The episode-level form, with the total return `sum_t G_t` in the numerator,
looks equally defensible from the derivation, since the expectation there is
over trajectories. It is not interchangeable, and the failure is quantitative
rather than subtle: on this gridworld `sum_t G_t` is around -400 while `G_t`
is around -8, so `b` lands an order of magnitude away from the quantity it is
subtracted from, every advantage becomes a large positive number, the policy
is pushed towards every action it happened to take, and the expected return
falls to -85. Verified by running it.

## Results on the gridworld

10 seeds, 100,000 steps, step size 0.01. The optimum is 0.941.

| baseline | mean final | median final | std | seeds solved | mean over training |
|---|---|---|---|---|---|
| none | 0.847 | 0.941 | 0.282 | 9 of 10 | **0.650** |
| mean | -0.045 | 0.941 | 2.958 | 9 of 10 | -0.310 |
| optimal | -0.045 | 0.941 | 2.958 | 9 of 10 | -0.507 |

All three solve nine seeds out of ten and have the same median. The means and
standard deviations differ only because of which single seed went worst: for
`none` the odd seed out ends at 0, and for the two baselines it ends on a
penalty tile at about -9. Averaging ten values drawn from `{0.941, 0, -9}`
produces a number that is mostly a report on one run, which is why the median
and the solved count are given alongside it.

**So at this step size the three baselines are indistinguishable.** That is
worth saying plainly rather than manufacturing a ranking out of the means.

## Where the baseline actually shows up

Variance reduction does not show up as a better final answer when the step
size is small enough for everything to work. It shows up as tolerance of a
larger step size. Final return after 60,000 steps, 3 seeds:

| baseline | a = 0.003 | a = 0.01 | a = 0.03 | a = 0.1 |
|---|---|---|---|---|
| none | 0.941 | 0.628 | 0.628 | -2.659 |
| mean | 0.941 | **0.941** | **0.941** | **0.628** |
| optimal | 0.941 | **0.941** | 0.628 | -2.659 |

The mean baseline keeps working at a step size ten times larger than the one
at which `none` first degrades, and it is the only variant still producing a
positive return at `a = 0.1`. The optimal baseline sits between the two,
which is a useful reminder that "variance minimising" is a statement about a
single gradient estimate, not a guarantee about the whole optimisation: it
minimises the variance of the estimator given the current policy, and says
nothing about the trajectory the optimiser then takes.

The failure mode at large step sizes is specific. With no baseline every
return on this gridworld is large and negative, so the first update pushes
*every* action's preference down by a similar large amount, the softmax
saturates, `grad log pi` goes to zero, and learning stops permanently after
one update. Dividing the gradient by the episode length hides this, because
it makes the effective step independent of episode length, but that is no
longer the REINFORCE estimator. The implementation here keeps the correct sum
over time steps and uses a smaller step size instead, and the table above is
why the step size is what it is.

## Results on Pendulum

5 seeds, 200,000 steps, Gaussian policy with `sigma` decaying from 2.0 to 0.1.

| baseline | mean final | median final | std | mean over training |
|---|---|---|---|---|
| none | **-402.7** | **-368.2** | 212.9 | -419.5 |
| mean | -532.3 | -499.3 | **75.6** | -512.3 |
| optimal | -528.0 | -511.4 | **62.8** | -526.3 |

Two honest observations.

**None of these solves Pendulum.** The usual threshold is about -150 and the
best run here is around -370. Linear REINFORCE over RBF features, with a
scheduled rather than learned standard deviation and no critic, is simply not
a strong enough method for this problem at this budget. Reporting -400 as a
result rather than as a success is the point.

**The baselines cut the spread across seeds by a factor of three** (212.9
down to 62.8) while making the mean worse. That is a clean separation of the
two things a baseline does and does not do: it reduces the variance of the
gradient estimate, which shows up as runs agreeing with each other, and it
does not by itself move the optimum the method converges towards. Here all
three appear to be converging into the same mediocre basin and the baselines
simply get there more reliably.

## A note on the Gaussian parameterisation

`sigma` is the standard deviation everywhere. `np.random.normal` takes the
standard deviation, and the log gradient divides by the *variance*. Passing
`sigma ** 2` to the sampler while still dividing by `sigma ** 2` in the
gradient leaves it off by a factor of `sigma ** 2`, which here would be 4 at
the start of training and 0.01 at the end: an uncontrolled learning rate
schedule hiding inside the annealing, and invisible in any plot.
`test_dlog_gaussian_matches_a_numerical_gradient` checks the gradient against
finite differences for that reason.

One module runs both the discrete and the continuous problem from a
configuration dictionary, rather than carrying two setups with one commented
out, and the policy functions live in `rlkit.policies` where the tests can
reach them.

## Figures

| file | content |
|---|---|
| `figures/reinforce_gridworld.png` | the three baselines, 10 seeds, with confidence bands |
| `figures/reinforce_step_size.png` | how large a step size each baseline tolerates |
| `figures/reinforce_pendulum.png` | the three baselines on Pendulum, 5 seeds |
| `results/reinforce_summary.csv` | the tables above |
| `results/*_curves.npy` | the per seed curves, stored at one point per 100 steps in float32 so figures can be redrawn without re-running |
