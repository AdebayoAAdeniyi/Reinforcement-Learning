# 08. Actor-critic: n-step and lambda-return critics

**Book:** Sutton and Barto, 2nd edition. The n-step return is section 7.1 and
equation 7.1, the lambda-return is section 12.1 and equation 12.2,
actor-critic with eligibility traces is the box in section 13.5 and the one
step version is equation 13.17.
**Environment:** `Gym-Gridworlds/Penalty-3x3-v0`, `gamma = 0.99`, softmax
policy over 49 RBF features, actor step 0.01, critic step 0.05, 10 episodes
per update, 150,000 steps, 5 seeds.

```bash
python actor_critic.py
```

## What this adds to chapter 07

REINFORCE weights each log gradient by the full Monte Carlo return `G_t`,
which is unbiased and high variance. An actor-critic replaces that with an
advantage built from a learned value function. Three critics are compared,
which is exactly the bias and variance axis:

* `mc`, the full return. No bias, most variance. This is REINFORCE with a
  learned state value baseline, and it is the reference point.
* `nstep`, the n-step return. `n` sets the trade-off directly: `n = 1`
  bootstraps hardest, large `n` approaches Monte Carlo.
* `lambda`, the lambda-return, a geometric average over all n-step returns,
  computed backwards in its recursive form
  `G_lam(t) = r + gamma[(1 - lam) V(s') + lam G_lam(t+1)]`.

## Results

All three reach the optimal return of 0.941 on every seed, so the final value
says nothing. What separates them is how fast they get there, measured as the
mean expected return over the whole of training:

| critic | final return | mean over training |
|---|---|---|
| Monte Carlo return | 0.941 | **0.833** |
| lambda-return, `lambda = 0.9` | 0.941 | 0.788 |
| 5-step return | 0.941 | 0.540 |

and the effect of `n` is monotone:

| n | final return | mean over training |
|---|---|---|
| 1 | 0.941 | 0.257 |
| 3 | 0.941 | 0.394 |
| 5 | 0.941 | 0.540 |
| 20 | 0.941 | **0.819** |

### More bootstrapping is worse here, and that is not the usual story

The standard argument for a critic is that bootstrapping cuts the variance of
the gradient estimate and therefore speeds learning. On this problem the
ranking runs the other way, cleanly and monotonically: `n = 1` is the slowest
and `n = 20`, which is nearly Monte Carlo on episodes of about seven steps,
is the fastest.

Two features of this environment explain it, and both are worth being
explicit about because they are also the conditions under which the standard
argument *does* hold elsewhere.

First, there is very little variance to remove. The gridworld is
deterministic, the reward has no noise, and episodes are short. The Monte
Carlo return from a state is therefore almost a fixed number rather than a
high variance sample, so the variance reduction a critic offers is close to
nothing.

Second, the bias is real and large early on. The critic starts at zero and is
learning at the same time as the actor. A one-step target is
`r + gamma * V(s')` where `V` is, for the first several thousand steps, mostly
wrong. The actor is therefore being pushed by a confidently wrong advantage.
Monte Carlo has no such problem: its target never consults the critic for
anything except the baseline, which can be wrong without changing the
direction of the update, only its magnitude.

The lambda-return at `lambda = 0.9` sits at 0.788, close to Monte Carlo's
0.833, which is what the geometry predicts: at `lambda = 0.9` the weight on
the n-step return decays as `0.9^(n-1)`, so most of the mass is on long
horizons. It is a reasonable default precisely because it does not have to
commit to a single `n`.

The honest generalisation is not "critics do not help". It is that the
benefit of a critic scales with the variance of the return, which scales with
episode length, reward noise and environment stochasticity, and none of those
are present here. Chapter 03 showed the same quantity from the other side:
adding reward noise of standard deviation 3 made Monte Carlo estimates
visibly worse, and that is exactly the regime where a critic earns its bias.

## A negative result: DangerMaze-6x6

None of the three critics solves `DangerMaze-6x6-v0` within 60,000 steps. That is reported
rather than dropped, with the training reward alongside the evaluation
return so the reader can see what actually happens:

| critic | evaluation return (sparse) | training reward per episode (shaped) |
|---|---|---|
| mc | 0.000 | -102.2 → -8.70 |
| nstep | 0.000 | -102.2 → -8.80 |
| lambda | 0.000 | -102.2 → -8.93 |

The agent learns to stop falling into the pits, which takes the shaped
training reward from about -102 to about -9, and then stops improving. Two
things are happening. Avoiding a -100 penalty and reaching a +1 goal are
different problems, and the first dominates the gradient by two orders of
magnitude. And once the softmax preferences grow large enough that the policy
is effectively deterministic, `grad log pi` goes to zero and learning stops
whether or not the policy is any good. `figures/actor_critic_dangermaze.png`
shows both panels.

What would be needed is not a different critic. It is an exploration
mechanism that survives the policy becoming confident: an entropy bonus, a
temperature that does not collapse, or a reward scale that does not let one
term dominate.

## Implementation notes

The three critics differ in exactly one function, `advantages`, which turns a
trajectory into per-step advantages and critic targets. The rollout, the
actor update and the critic update are shared, so a difference in the curves
is a difference in the critic and nothing else.

The bootstrap convention in that function is where the care goes: `V[T]` is
zero when the episode terminated and the critic's own estimate when it was
truncated by the time limit. That distinction matters more in an
actor-critic than anywhere else in the repository, because the critic's
target *is* the algorithm. Treating a truncation as a termination teaches the
agent that the world ends at `max_episode_steps`.

`DangerMaze-6x6-v0` exists in the pinned `gym_gridworlds` and was renamed
upstream afterwards, so `rlkit.envs.make_env` resolves it and reports clearly
if it cannot. See
[`docs/ENGINEERING_NOTES.md`](../docs/ENGINEERING_NOTES.md) section 1.

## Figures

| file | content |
|---|---|
| `figures/actor_critic_comparison.png` | the three critics on Penalty-3x3 |
| `figures/actor_critic_nstep.png` | the effect of `n` |
| `figures/actor_critic_dangermaze.png` | the negative result, evaluation return and training reward |
| `results/actor_critic_summary.csv` | the tables above |
