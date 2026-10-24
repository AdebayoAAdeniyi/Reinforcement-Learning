# Engineering notes

How this repository is built so that its numbers can be trusted, and the
reasoning behind the decisions that are not obvious from the code. Written
for a reader who wants to know whether the results mean anything before
reading the results.

---

## 1. Pinning a research dependency, and surviving the pin being lifted

Every tabular chapter builds on `gym_gridworlds` by Simone Parisi. It is a
research package under active development, and between the version these
experiments were written against and the current one it changed in two ways
that do **not** raise an error:

| | October 2024 (`250e628`) | current (`c6d4066`) |
|---|---|---|
| action encoding | LEFT 0, **DOWN 1, RIGHT 2**, UP 3, STAY 4 | LEFT 0, **RIGHT 1, DOWN 2**, UP 3, STAY 4 |
| `Empty-2x2-v0` | registered | renamed `Empty-RandomStart-2x2-v0` |
| `DangerMaze-6x6-v0` | registered | replaced by `DangerMaze-5x6-v0` |

The renames fail loudly. The action reordering does not: a policy written as
`[[1, 2, 4], [1, 2, 3], [2, 2, 3]]` is a different policy under the two
versions and nothing anywhere reports a problem. The agent simply learns a
different task.

Two defences, because a pin alone is brittle:

**The pin.** `requirements.txt` specifies the exact commit, so the published
numbers reproduce.

**Resolution at import time.** `rlkit/envs.py` reads the action constants out
of whatever version is installed and exposes `policy_from_names`, so a policy
is written as

```python
OPTIMAL_POLICY_NAMES = [
    ["DOWN",  "RIGHT", "STAY"],
    ["DOWN",  "RIGHT", "UP"],
    ["RIGHT", "RIGHT", "UP"],
]
```

and means the same thing under either encoding. `resolve_env_id` does the
same for environment ids, mapping a removed id to its replacement and raising
a message that names the version problem if neither exists.

The general lesson, and the reason this is the first section: a silent
semantic change in a dependency is far more dangerous than a crash, and the
fix is to stop encoding meaning in integers that the dependency owns.

---

## 2. Termination belongs to the transition, not to the next state

The Bellman backup used throughout is

```
v(s) = sum_a pi(a|s) [ R[s,a] + (1 - T[s,a]) * gamma * sum_s' P[s,a,s'] v(s') ]
```

`T[s, a]` is 1 when the transition *leaving* `(s, a)` ends the episode. In
`Penalty-3x3` the only such transition is `(state 2, STAY)`.

It is easy to write `T[s_next, a]` instead, which asks whether the next state
would terminate under the same action index. That is a different quantity, it
is nonzero in states the agent is merely passing through, and it moves the
fixed point of the operator without producing anything that looks wrong in a
figure.

Termination is applied explicitly inside each operator in `rlkit/bellman.py`
rather than folded into `P` during extraction, so the indexing is visible at
the point where it matters.

Two tests pin this down:

* `test_v_and_q_operators_agree` checks `v_pi(s) = sum_a pi(a|s) q_pi(s,a)`
  for random policies. Two operators that disagree on that identity cannot
  both be right, and a misindexed termination factor breaks it.
* `test_terminal_transition_does_not_bootstrap` checks
  `q_*(goal, STAY) = 1.0` to within `1e-12`. The terminal transition must be
  worth exactly its reward, not its reward plus a discounted continuation.

---

## 3. Reproducibility, and the part of it that is not obvious

Every learning routine takes an explicit `numpy.random.Generator` built from
its `seed` argument. No global `np.random` state, and no standard-library
`random`, whose generator `np.random.seed` does not affect.

That is necessary and it is **not sufficient**. `env.reset(seed=...)` seeds
the environment's own generator but not the generator belonging to
`env.action_space`. They are separate objects, and the space's generator is
initialised from operating-system entropy when the space is constructed.

That is normally harmless, but `gym_gridworlds` implements its stochastic
transitions like this:

```python
if self.np_random.random() < self.random_action_prob:
    action = self.action_space.sample()
```

So in chapter 06, where `random_action_prob = 0.1`, one action in ten came
from a generator nothing had seeded, and a sweep that looked fully
deterministic was not. It was caught by running the same sweep twice and
comparing: ten of twenty-seven cells differed, some by a wide margin.

Demonstrated directly, three identical rollouts with
`random_action_prob = 0.5`, hashing observations and rewards:

```
env.reset(seed=k) only:        453bb9343afa  7152e890117c  01db8d05fe5a
rlkit.envs.reset_seeded(env,k): 35b14fae50b1  35b14fae50b1  35b14fae50b1
```

`rlkit.envs.reset_seeded` seeds the environment, the action space and the
observation space together, and is used at the start of every run and every
evaluation episode in every chapter.
`test_reset_seeded_makes_a_stochastic_environment_reproducible` fails without
it. After the fix, two separate processes produce byte-identical results for
the same configuration.

This is the clearest argument in the repository for having a test suite at
all: the problem was invisible in every figure.

---

## 4. Terminated and truncated are different, and the distinction is in the target

`terminated` means the MDP ended: there is no next state, and the TD target
is the reward alone. `truncated` means a time limit cut off an episode that
was still running: the target must still bootstrap, or the agent is taught
that the world ends after `max_episode_steps`.

Collapsing them into one `done` flag and bootstrapping in both cases corrupts
exactly the transitions that carry the reward signal. In `Penalty-3x3` there
is one terminal transition and it is the only place the `+1` enters.

The distinction matters most in chapter 08, where the critic's bootstrap is
the whole algorithm: an episode cut off by the limit bootstraps from the last
state, an episode that genuinely ended does not.

---

## 5. Counting, so that a comparison means something

Chapter 02 compares three algorithms by the number of policy-evaluation
steps. That number is meaningless unless all three count the same event, so
**one step is one Bellman backup applied to the whole state space**, and
exactly one Bellman error is logged per step, for all three algorithms and
for both `V` and `Q`.

Two related decisions follow from the same concern:

* All three use a **synchronous** sweep, computing the whole new value
  function from the whole old one. In-place Gauss-Seidel updates converge
  faster, so mixing the two between algorithms would make the comparison
  partly a comparison of update style.
* GPI stops only when the policy is stable **and** the last evaluation sweep
  is below tolerance. Stopping on policy stability alone would let it finish
  with an unconverged value function, which would flatter its iteration count
  for a reason unrelated to the algorithm.

The chapter then reports the tolerance sensitivity as a result in its own
right, because value iteration and GPI reach an exactly zero Bellman error
here while policy iteration converges only geometrically. "PI is N times
slower" is not a well-formed claim without naming `theta`.

---

## 6. Choosing a metric that can answer the question

Several chapters log more than one metric because the obvious one misleads.

**Chapter 03.** The Bellman error is measured against `q_pi` of a policy that
is itself changing as `epsilon` decays, so the reference moves: as the policy
becomes near-greedy, `q_pi` grows extreme in states that policy would loiter
in, and the logged error can rise while the estimates improve. Paired with
it is the exact value of the start state under the greedy policy, computed
from the model, which is a fixed reference and answers a different and more
important question: is the agent any good yet.

For the off-policy runs the reference is `q_pi` of the **target** policy,
since that is what off-policy MC estimates. Scoring both runs against the
behaviour policy would measure each estimator against something it is not
trying to estimate.

**Chapter 04.** TD error, Bellman error and expected return are all logged,
and the chapter is built around showing them disagree. State visitation is
logged too, because the effect of the initial value turns out to be an
exploration effect and a visit histogram is evidence rather than assertion.

**Chapter 06.** The TD error is reported alongside the return precisely
because the settings with the lowest TD error have the worst returns.

---

## 7. Designing a controlled comparison

Two choices in chapter 06 are worth stating because the obvious version of
each would have produced a confident, meaningless answer.

**The replay buffer has to be much larger than the batch.** Comparing
"consecutive batch of D" against "sample D from a buffer of the last D" draws
both from the same window; the two differ only in which samples repeat. That
is a bootstrap resample, not an independence test. `BUFFER_SIZE` is the whole
run.

**At D = 1 the two modes are not the same algorithm.** It is tempting to
treat a batch of one as a control condition, but `on_the_fly` fits the
transition that just happened while `replay` fits one drawn from anywhere in
the history. The chapter says so rather than reading the D = 1 column as a
null result.

A third, in chapter 05: comparing feature constructions with as many centres
as data points gives the local methods one free parameter per sample, so they
interpolate exactly and score zero on any target. The chapter runs the sweep
at 20 centres and at 100 and reports both, because the degenerate case is
itself the lesson.

---

## 8. Step sizes that let the comparison be about the method

Chapter 05 divides the gradient-descent step size by the largest eigenvalue
of the feature Gram matrix. Without it, one `alpha` cannot serve all five
feature constructions: a polynomial basis on `[-10, 10]` has features of
magnitude `10^5` while tile coding has features of magnitude 1, so a step
size that converges for one diverges for the other. Hand-tuning `alpha` per
method would make part of any reported difference a difference in tuning
effort.

Chapter 07 makes the opposite move deliberately. At a step size small enough
for everything to work, all three REINFORCE baselines solve nine seeds out of
ten and the comparison says nothing, so the chapter sweeps the step size and
reports how large a one each baseline tolerates. That is where variance
reduction actually shows up.

---

## 9. Small decisions that change results

**Tie-breaking in epsilon-greedy.** `np.argmax` gives the whole `1 - eps` to
the lowest index even when several actions are equally good. Several actions
in these gridworlds bump into a wall and are genuinely identical, so that
biases exploration. `rlkit.policies.eps_greedy_probs` shares the greedy mass
among tied actions.

**Expected rewards, not noisy samples.** The tables `R`, `P` and `T` are
always extracted from an environment built **without** reward noise. `R[s,a]`
must be the expected reward, and one noisy sample per state-action pair is
not that.

**Reaching `set_state` through the wrapper.** `gymnasium.make` returns a
`TimeLimit`/`OrderEnforcing` wrapper that does not expose `set_state`, so MDP
extraction goes through `env.unwrapped`. With `coordinate_observation=True`
the same method takes `(row, col)` coordinates rather than a flat index.

**Headless plotting.** `rlkit/plotting.py` forces the Agg backend and every
script writes figures to disk. `plt.ion()` and `plt.show()` draw nothing on a
machine without a display, which includes continuous integration.

**Log axes and exact zeros.** Several Bellman error curves reach exactly
zero, which a log axis cannot draw. Those points are clipped to half the
smallest positive value so the final point stays visible, and the true
iteration count is printed in the panel.

---

## 10. Scope and provenance

Stated plainly so that nobody compares numbers that are not comparable.

**The chapter 05 dataset is generated by the repository, not shipped.**
`05_function_approximation/make_dataset.py` builds it from a pinned
environment with one command, so its construction is visible and adjustable
rather than opaque, and the chapter reproduces from scratch.
`DangerMaze-6x6-v0` was chosen because its value function has real
discontinuities, from `-105` next to `-0.04`; an empty grid gives a smooth
surface that every feature construction fits almost perfectly and so cannot
discriminate between them. Choosing an environment that can tell the methods
apart is part of the experiment design.

**The two chapter 05 regression targets are chosen for contrast**, not drawn
from a benchmark: `sin(x) + 0.1 x^2` is smooth and nearly polynomial, and the
piecewise target has two jump discontinuities that no low-order polynomial
can express. The pair separates "the optimiser needs more steps" from "the
representation cannot express the answer".

**Chapter 08 has no assignment specification**, so it is built from the book:
n-step returns (section 7.1), the lambda-return (section 12.1) and
actor-critic (section 13.5).

---

## 11. Honest limits of the results

Three experiments do not solve their task, and the repository reports that
with the mechanism rather than quietly dropping the run. Negative results
that come with an explanation are evidence; negative results that are hidden
are a liability.

* **Chapter 03.** No configuration produces a policy that reaches the goal.
  This is a property of on-policy MC control, not of the implementation:
  section 5.4 says it converges to the best epsilon-soft policy, and the
  chapter shows the exact `q_pi` values at which the greedy action flips.
  Checked against longer training, a constant step size, a larger `gamma` and
  a longer horizon.
* **Chapter 07.** REINFORCE on Pendulum plateaus near `-370` to `-530`,
  short of the `-150` usually taken as solved. A linear approximator over RBF
  features with a scheduled standard deviation and no critic is not a strong
  enough method at this budget.
* **Chapter 08.** None of the three critics solves `DangerMaze-6x6-v0` within
  the budget. All three take the shaped training reward from `-102` to about
  `-9`, which is the agent learning to avoid the pits and then stopping,
  because the softmax saturates and `grad log pi` goes to zero. What that
  calls for is an exploration mechanism, not a different critic.
