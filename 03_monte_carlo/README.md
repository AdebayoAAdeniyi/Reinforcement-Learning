# 03. Monte Carlo control, on-policy and off-policy

**Book:** Sutton and Barto, 2nd edition, chapter 5. Every-visit MC is section
5.1, on-policy epsilon-soft control is the box in section 5.4, weighted
importance sampling is equations 5.6 and 5.8, off-policy MC control is the
box in section 5.7.
**Environment:** `Gym-Gridworlds/Penalty-3x3-v0`, `gamma = 0.9`, episode
horizon 10, budget 2000 environment steps, 50 seeds.

```bash
python monte_carlo_control.py
```

This is the first chapter where the model is unknown and the agent has to
learn from sampled returns, which brings in both the variance problem and
the on-policy versus off-policy distinction.

## Weighted importance sampling

Walking a trajectory backwards from its last step, with behaviour policy `b`
and target policy `pi`:

```
G      <- r_t + gamma * G
C(s,a) <- C(s,a) + W
Q(s,a) <- Q(s,a) + (W / C(s,a)) * (G - Q(s,a))
W      <- W * pi(a|s) / b(a|s)
if W == 0: break
```

Three details decide whether that is weighted importance sampling or an
expensive way of computing a plain average:

1. `C` is a **cumulative** weight that persists across episodes, so it is
   created once outside the episode loop. Rebuilt per episode it resets to
   zero, the step size `W / C` becomes 1, and all history is discarded.
2. `W` accumulates along the trajectory as a product of per-step ratios and
   is updated **after** it is used. A single-step ratio recomputed from
   scratch is not an importance sampling ratio.
3. `b(a|s)` is recorded at the moment the action was taken, not recomputed
   later from an already-updated `Q`, which would give the ratio of two
   different behaviour policies.

## What this chapter measures, and why it logs two things

Two metrics are logged. The first is the total absolute Bellman error against
`q_pi`. The second is a fixed reference: the exact value of the start state,
computed from the true model, under the policy that is greedy with respect to
the current `Q`. Its maximum is `v_*(s0) = 0.9^6 = 0.531`.

The second exists because the first has a moving reference. As `epsilon`
decays, `q_pi` of the near-greedy policy grows extreme in states that policy
would loiter in, so the logged error can rise while the estimates improve.
`figures/mc_on_vs_off.png` shows the on-policy curve falling until about step
600 and then climbing for the rest of the run, which is mostly the reference
moving.

For the off-policy runs the reference is `q_pi` of the **target** policy,
since that is what off-policy MC estimates. Scoring both against the
behaviour policy would measure the off-policy estimator against something it
is not trying to estimate.

## The headline result: the greedy policy never reaches the goal

The fixed-reference metric comes out as **0.000** for every one of the 36
on-policy settings. That is not an implementation failure. It is what
on-policy Monte Carlo control is supposed to do here, and it is the most
instructive thing in the chapter.

Sutton and Barto section 5.4 says MC control with an epsilon-soft policy
converges to the best policy *among the epsilon-soft policies*, which is not
the optimal policy. On this gridworld that gap changes the greedy action.
Here is `q_pi` at state 6, the bottom-left corner, for the epsilon-soft
policy built around the optimal policy, computed exactly from the model:

| epsilon | LEFT | DOWN | RIGHT | UP | STAY | greedy action |
|---|---|---|---|---|---|---|
| 0.0 | 0.59 | 0.59 | **0.66** | 0.53 | 0.59 | RIGHT |
| 0.1 | 0.15 | 0.15 | **0.17** | -0.04 | 0.15 | RIGHT |
| 0.3 | **-1.05** | -1.05 | -1.16 | -1.58 | -1.05 | LEFT |
| 0.5 | -2.87 | **-2.87** | -3.24 | -3.86 | -2.87 | DOWN |
| 0.8 | **-7.37** | -7.37 | -8.63 | -9.65 | -7.37 | LEFT |

`RIGHT` is the optimal action at state 6, and it stays the greedy action only
while `epsilon` is at most about 0.2. Above that, walking into a wall is
genuinely better *under that policy*, because moving towards the goal means
passing near the two tiles worth `-10` while still taking random actions a
third of the time. The agent is not confused; it is correctly evaluating a
policy too exploratory to be worth following.

The schedule here starts at `epsilon = 1.0` and reaches 0.01 only at the end
of the 2000-step budget, and the every-visit average weights all that early
experience equally. Running for 50,000 steps does not change it, and neither
does a constant step size in place of the running average. Both were tried.

Three things follow, and they are the real content of the chapter:

* **Q-learning in [chapter 04](../04_temporal_difference/) solves the same
  gridworld in 10,000 steps**, because it is off-policy: its `max` target
  estimates `q_*` regardless of how exploratory the behaviour policy is.
  This is the clearest demonstration of the on-policy and off-policy
  distinction anywhere in the repository, and it falls out of two
  consecutive chapters rather than being asserted.
* **Off-policy MC with a greedier target does better**, which is consistent:
  at 1 episode per update it reaches `v(s0)` of 0.021 and 0.043 at decays 2
  and 5, against exactly 0.000 for every on-policy setting. Small, but it is
  the only thing in the sweep that is not zero.
* **The hyperparameter tables below rank value-estimation accuracy, not task
  success.** That is still the thing the algorithm is for. It is not evidence
  that one setting learns to solve the gridworld.

## Hyperparameter results

Total absolute Bellman error after 2000 environment steps, mean over 50
seeds. Lower is better.

**On-policy**

| episodes per update | decay 1 | decay 2 | decay 5 |
|---|---|---|---|
| 1, no noise | **122.3** | 126.1 | 139.0 |
| 10, no noise | 466.4 | 490.4 | 507.7 |
| 50, no noise | 561.5 | 620.5 | 736.5 |
| 1, noise 3.0 | **139.9** | 153.7 | 178.8 |
| 10, noise 3.0 | 479.6 | 509.6 | 520.5 |
| 50, noise 3.0 | 574.3 | 637.6 | 741.7 |

**Off-policy, weighted importance sampling, target epsilon 0.1**

| episodes per update | decay 1 | decay 2 | decay 5 |
|---|---|---|---|
| 1, no noise | **39.9** | 60.6 | 103.9 |
| 10, no noise | 368.5 | 450.6 | 506.4 |
| 50, no noise | 706.2 | 723.2 | 729.6 |
| 1, noise 3.0 | **102.8** | 131.2 | 173.1 |
| 10, noise 3.0 | 402.6 | 501.0 | 544.8 |
| 50, noise 3.0 | 710.4 | 723.0 | 729.0 |

### Episodes per update dominates, and more is worse

Going from 1 to 50 episodes per update multiplies the final error by about
four and a half; going from decay 1 to decay 5 multiplies it by about 1.1 to
1.3.

This is a budget-allocation result rather than a statistical one. Collecting
more episodes per update buys a lower-variance return estimate for each
state-action pair. It costs updates: the budget is 2000 environment steps, so
at roughly 7 steps per episode, 1 episode per update gives about 280 updates
and 50 episodes per update gives about 5. Five updates cannot move `Q` near
`q_pi` from a zero initialisation, whatever the quality of each one.

The gap narrows slightly when reward noise is added, which is the expected
direction: noise makes each individual return less informative, which is
exactly the situation averaging is meant to help, so the penalty for fewer
updates is a little less severe.

### What a faster decay does

Decay 5 drives epsilon to its floor five times sooner, so the behaviour
policy commits early on very little data. The error is consistently worse at
every setting, and the margin is largest where updates are frequent enough
for the committed policy to matter. The problem is small, nine states, so one
might expect to decay fast and still find the optimum, but the two tiles
worth `-10` make early commitment costly.

### Did importance sampling help

It depends on the update frequency, which is a more interesting answer than
yes or no.

* At 1 episode per update it helps substantially: 39.9 against 122.3 without
  reward noise. The target policy is greedier than the behaviour policy, so
  the estimate converges towards a better policy while the behaviour policy
  keeps exploring.
* At 50 episodes per update it hurts: 706.2 against 561.5. With so few
  updates the cumulative weight `C` has barely accumulated, so the step size
  `W / C` is still large and erratic, and the `W == 0` break discards the
  prefix of every trajectory in which the behaviour policy took an action the
  target policy would not have. Off-policy MC throws data away, and it can
  only afford to when there is plenty of it per update.

Reward noise widens the gap in the unhelpful direction at the frequent-update
setting, 102.8 against 39.9. Weighted importance sampling reduces variance
compared with ordinary importance sampling but does not eliminate it, and
noisy returns multiplied by a ratio are noisier still.

### Would a random initial state help, and is that exploring starts

It would help, because here it is the only mechanism available for reaching
state-action pairs the current policy avoids. It is not the same as exploring
starts, which requires every state **action** pair to have nonzero
probability of beginning an episode. Randomising only the state still lets
the policy choose the first action, so pairs the policy never selects are
still never sampled from a start, and the convergence argument in section 5.3
needs the stronger condition.

### What stochastic rewards expose about MC

Variance. A Monte Carlo update uses the whole return from a state to the end
of the episode, so every reward along the way contributes its noise. With a
horizon of 10 and noise of standard deviation 3 on each reward, the return
has a standard deviation of roughly 3 times the square root of the remaining
steps, several times larger than the `+1` the agent is trying to detect. TD
methods in the next chapter bootstrap after one step and absorb one reward's
worth of noise instead of ten.

## Figures

| file | content |
|---|---|
| `figures/mc_on_policy.png` | the nine-setting sweep, Bellman error above and the fixed-reference metric below |
| `figures/mc_importance.png` | the same sweep off-policy with weighted IS |
| `figures/mc_on_vs_off.png` | the two head to head at 10 episodes, decay 1 |
| `results/final_bellman_error.csv` | the tables above, including `final_greedy_v0` |
