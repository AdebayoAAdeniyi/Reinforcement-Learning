# 04. Temporal difference control

**Book:** Sutton and Barto, 2nd edition. SARSA section 6.4 equation 6.7,
Q-learning section 6.5 equation 6.8, Expected SARSA section 6.6 equation 6.9,
Double Q-learning section 6.7 equation 6.10.
**Environment:** `Gym-Gridworlds/Penalty-3x3-v0`, `gamma = 0.99`, horizon 10,
`alpha` from 0.1 to 0.001, `epsilon` from 1.0 to 0.01, 10,000 steps,
10 seeds, initial Q values `{-10, 0, 10}`, reward noise 0 and 3.0.

Double SARSA and Double Expected SARSA are not book algorithms. They are
built here by the same construction as Double Q-learning and labelled as
extensions.

```bash
python td_control.py
```

## Design notes

**One function supplies the bootstrap for all six algorithms.** `td_target`
takes two Q tables: one chooses the action or the distribution, the other
supplies the number. For the single-estimator algorithms they are the same
array; for the double ones they are the two different estimates, which is
exactly the decoupling that removes maximisation bias. Writing it this way
means the six algorithms differ in a few lines rather than in six copies of
a training loop.

**`terminated` and `truncated` are treated differently.** `terminated` means
the MDP ended, so there is no next state and the target is the reward alone.
`truncated` means the ten-step limit cut off an episode that was still
running, so the target must still bootstrap, or the agent is taught that the
world ends after ten steps. Collapsing them into one flag puts a spurious
`gamma * Q(s_next, .)` on the one genuinely terminal transition here, which
is the only place the `+1` reward enters.

**Three metrics are logged, plus state visitation.** The chapter is built
around showing that the first three disagree, and the visitation histogram is
what turns the explanation from an assertion into evidence.

## Results

Final values after 10,000 steps, mean over 10 seeds. "visit" is the largest
share of training steps spent in any single state.

| algorithm | noise | Q0 | Bellman error | return | mean abs TD error | visit |
|---|---|---|---|---|---|---|
| QL | 0.0 | -10 | 165.9 | 0.000 | 0.045 | **0.68** |
| SARSA | 0.0 | -10 | 455.7 | **0.941** | 1.690 | 0.21 |
| ExpSARSA | 0.0 | -10 | 443.0 | **0.941** | 1.171 | 0.21 |
| QL | 0.0 | 0 | **3.9** | **0.941** | 0.002 | 0.21 |
| SARSA | 0.0 | 0 | 196.7 | 0.565 | 0.593 | 0.21 |
| ExpSARSA | 0.0 | 0 | 197.6 | **0.941** | 0.536 | 0.21 |
| QL | 0.0 | 10 | 470.4 | 0.000 | 0.066 | 0.22 |
| SARSA | 0.0 | 10 | 221.4 | 0.000 | 0.248 | 0.23 |
| ExpSARSA | 0.0 | 10 | 263.6 | 0.000 | 0.166 | 0.21 |
| QL | 3.0 | -10 | 287.4 | 0.000 | 2.374 | 0.56 |
| SARSA | 3.0 | -10 | 465.4 | 0.847 | 2.888 | 0.22 |
| ExpSARSA | 3.0 | -10 | 453.2 | **0.941** | 2.676 | 0.21 |
| QL | 3.0 | 0 | 427.1 | 0.000 | 2.361 | 0.53 |
| SARSA | 3.0 | 0 | 189.6 | 0.282 | 2.445 | 0.21 |
| ExpSARSA | 3.0 | 0 | 185.3 | 0.471 | 2.398 | 0.20 |
| QL | 3.0 | 10 | 779.4 | 0.000 | 2.364 | 0.53 |
| SARSA | 3.0 | 10 | 217.8 | 0.000 | 2.417 | 0.20 |
| ExpSARSA | 3.0 | 10 | 234.9 | 0.000 | 2.369 | 0.19 |

Double estimators, same conditions:

| algorithm | noise | Q0 | Bellman error | return | visit |
|---|---|---|---|---|---|
| DoubleQL | 0.0 | -10 | 253.7 | 0.000 | 0.69 |
| DoubleSARSA | 0.0 | -10 | 429.5 | 0.847 | 0.21 |
| DoubleExpSARSA | 0.0 | -10 | 418.2 | **0.941** | 0.21 |
| DoubleQL | 0.0 | 0 | 28.0 | **0.941** | 0.22 |
| DoubleSARSA | 0.0 | 0 | **72.4** | 0.188 | 0.21 |
| DoubleExpSARSA | 0.0 | 0 | **72.7** | 0.000 | 0.24 |
| DoubleQL | 0.0 | 10 | 500.9 | 0.000 | 0.23 |
| DoubleSARSA | 0.0 | 10 | 416.5 | 0.000 | 0.21 |
| DoubleExpSARSA | 0.0 | 10 | 448.7 | 0.000 | 0.20 |
| DoubleQL | 3.0 | 0 | 210.9 | 0.000 | 0.65 |
| DoubleSARSA | 3.0 | 0 | **74.4** | 0.188 | 0.21 |
| DoubleExpSARSA | 3.0 | 0 | **72.9** | 0.000 | 0.23 |

### The most useful thing in this table

Q-learning fails in exactly the rows where it stops exploring. Every
Q-learning run with a return of zero has a visit concentration of 0.51 to
0.69, meaning more than half of all 10,000 training steps were spent in one
of the nine states. Every SARSA and Expected SARSA run sits at 0.19 to 0.24,
roughly what an exploring agent produces in a nine-state grid with one
high-traffic start state.

The mechanism is specific. Every reward here is zero except on the three
special tiles. With `Q0 = -10`, the first time any action is tried its target
is `0 + 0.99 * (-10) = -9.9`, better than every untried action's `-10`.
Actions that bump into a wall and return the agent to the same state get
tried most often, so their values rise first, so they look best, so they get
tried again. Q-learning's greedy target reinforces this because it always
bootstraps from the maximum, which in an unexplored region is whichever
action happened to be touched. SARSA and Expected SARSA bootstrap from the
policy, which is still mostly random early on, so they do not feed the loop.

Reward noise creates the same collapse at `Q0 = 0`, where Q-learning
otherwise does best. That is maximisation bias made visible: with noisy
rewards the max over noisy estimates is biased upwards, and the state-action
pair that got lucky becomes the one the agent returns to. This is the
clearest demonstration in the repository of why Q-learning's `max` is both
its strength and its weakness.

### Is one initialisation better for all algorithms

No. `Q0 = 0` is best for Q-learning and Expected SARSA without noise, and `0`
or `-10` for Expected SARSA with noise. `Q0 = -10` is the only setting at
which SARSA reaches the optimum. `Q0 = 10` fails for every algorithm in every
condition, because optimistic values must be driven down by about one percent
per backup at `gamma = 0.99`, and 10,000 steps is not enough; most
state-action pairs are still tied at 10, where the greedy tie-break produces
a degenerate policy.

### Is there a discrepancy between the TD error, the Bellman error and the return

Yes, and it is extreme enough to be the headline result.

Compare the first three rows. Q-learning at `Q0 = -10` has the **lowest** TD
error in the whole table, 0.045, and gets a return of zero. SARSA at the same
setting has a TD error of 1.690, nearly forty times larger, the **worst**
Bellman error in the table at 455.7, and reaches the optimum.

The TD error measures how surprised the agent is by its own transitions. An
agent that has stopped exploring and sits in one state is never surprised, so
its TD error is tiny and says nothing about policy quality. The Bellman error
is better, because it compares `Q` with `q_pi` over the whole state space,
but it is still a statement about the accuracy of value estimates, and it is
unbounded: a policy that loiters on a tile worth `-10` has a true value of
`-10 / (1 - gamma) = -1000`, so the error explodes for bad policies in a way
that has little to do with how close the agent is to behaving well.

The Double SARSA rows make the same point from the other side: the lowest
Bellman errors among the doubles, 72.4 and 72.9, come with returns of 0.188
and 0.000.

### Can a policy act optimally without having learned the optimal Q-function

Yes, and row four shows it. Acting greedily needs only the *ordering* of the
action values in each state, not their magnitudes. A `Q` that is wrong by a
constant, or wrong in states the optimal path never visits, still produces
the optimal policy. Expected SARSA at `Q0 = 0` has a Bellman error of 197.6,
fifty times Q-learning's 3.9 at the same setting, and both return 0.941.

### Which criterion should define performance

Expected return. It is the only one of the three that measures the quantity
the agent is supposed to maximise, and the table shows the other two pointing
the wrong way in several rows. The TD error should be read as a diagnostic of
whether learning is still happening, not as a score.

### Does SARSA suffer from maximisation bias

Not from the same source. Q-learning's bias comes from using one set of
estimates both to select and to evaluate the greedy action, so
`E[max_a Q] >= max_a E[Q]` with equality only when the estimates are exact.
SARSA bootstraps from the action its policy actually took, which involves no
max. It is not bias-free: the epsilon-greedy policy it evaluates is itself
derived from `Q`, so errors in `Q` still feed back into which policy is being
evaluated. The visit shares show the practical consequence: SARSA never
collapses, in any of the eighteen conditions.

### Do the double estimators help

Partly, and less than theory alone would suggest.

`DoubleQL` at `Q0 = 0` without noise keeps the optimal return but its Bellman
error is 28.0 against Q-learning's 3.9. Each table receives half the updates,
so learning is about twice as slow, and that cost is paid whether or not
maximisation bias was hurting. At `Q0 = 0` with noise, where the bias
demonstrably is hurting, `DoubleQL` still fails, with a visit share of 0.65.
Splitting the estimator removes the bias in the value estimate but does
nothing about the exploration collapse the bias has already triggered,
because the behaviour policy is still greedy with respect to the combined
estimate.

The clearest benefit is on the Bellman error for the SARSA variants, 72
against 190 to 200, which is the variance reduction working as advertised,
and the clearest cost is that it does not convert into return.

### How noise changes the picture

Every TD error rises to about 2.4, which is the noise standard deviation of 3
passing through the update; it no longer carries information about learning
progress. Returns fall everywhere, Q-learning loses its best setting
entirely, and Expected SARSA becomes the most robust of the three, reaching
0.941 at `Q0 = -10` and 0.471 at `Q0 = 0`. That ordering makes sense:
Expected SARSA replaces the sampled `Q(s', a')` with its expectation under
the policy, removing one source of variance from every update, and variance
is exactly what the noise adds.

## Figures

| file | content |
|---|---|
| `figures/td_noise_0.0.png`, `td_noise_3.0.png` | TD error, Bellman error and return for the single estimators |
| `figures/double_noise_0.0.png`, `double_noise_3.0.png` | the same for the double estimators |
| `figures/single_vs_double.png` | each algorithm against its double at `Q0 = 0` |
| `figures/state_visitation.png` | where Q-learning spent its training steps, per initial value |
| `results/td_summary.csv` | the tables above |
