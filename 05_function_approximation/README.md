# 05. Linear function approximation and semi-gradient TD

**Book:** Sutton and Barto, 2nd edition, chapter 9. Linear methods 9.4,
feature constructions 9.3 and 9.5, semi-gradient TD 9.3, the TD fixed point
9.4.

```bash
python features_demo.py      # part 1: the five constructions and their knobs
python regression_1d.py      # part 2: fitting a known 1D target
python make_dataset.py       # regenerate the gridworld dataset
python semi_gradient_td.py   # parts 3 and 4: TD prediction of v_pi and q_pi
```

## Part 1: the five constructions and their hyperparameters

| construction | hyperparameters | what they control |
|---|---|---|
| polynomial | degree | global curvature; every feature is nonzero everywhere, so a change anywhere moves the fit everywhere |
| RBF | centres, sigma | centres place the receptive fields, sigma sets how far each one generalises; activation is smooth and never exactly zero |
| tile coding | centres, width, offsets | width sets tile size, offsets set the number of tilings; the response is binary, so resolution comes only from having more tilings |
| coarse coding | centres, width, offsets | same geometry as tiles, but the response falls off linearly inside each field, so position within a field is still visible |
| state aggregation | centres | a one hot nearest centre lookup; the special case of coarse coding with no overlap and therefore no generalisation between groups |

`figures/features_hyperparams.png` varies one knob per row. The pattern that
matters for the rest of the chapter: widening sigma or the tile width buys
generalisation and loses resolution, and adding tilings recovers resolution
without narrowing anything, which is exactly why tile coding uses several
offset tilings rather than one fine one.

For a two dimensional state with very different ranges, say `[-10, 10]` by
`[0, 1000]`, the right change is per dimension widths rather than more
centres. A single scalar sigma makes the receptive fields enormously
elongated relative to one axis, so the approximator cannot resolve the first
dimension at all. Both `rbf_features` and `tile_features` here accept a width
per dimension for that reason.

## Part 2: fitting a known 1D target

Two targets on `[-10, 10]` with 100 samples, chosen to separate "the
optimiser needs more steps" from "the representation cannot express the
answer". `smooth` is `sin(x) + 0.1 x^2`, continuous and gently curved.
`piecewise` is built from cubics, an exponential, two flat sections and a
cosine burst, with two jump discontinuities.

**20 centres for 100 samples** (final mean squared error):

| construction | smooth | piecewise |
|---|---|---|
| polynomial, degree 5 | 2.149 | 21358.5 |
| RBF | 0.0091 | 2465.6 |
| tile coding, 3 tilings | 0.2207 | 2726.9 |
| coarse coding, 3 fields | **0.0207** | **1777.3** |
| state aggregation | 0.1526 | 2675.8 |

**100 centres for 100 samples**:

| construction | smooth | piecewise |
|---|---|---|
| polynomial, degree 5 | 2.149 | 21358.5 |
| RBF | 0.0010 | 280.6 |
| tile coding, 3 tilings | **0.0000** | **0.0000** |
| coarse coding, 3 fields | **0.0000** | **0.0000** |
| state aggregation | **0.0000** | **0.0000** |

The second table is the reason the first one exists. With one centre per
sample, tile coding, coarse coding and state aggregation each have a free
parameter per data point and interpolate exactly. Their mean squared error is
zero on **any** target whatsoever, so a ranking between them measures
nothing: the setup cannot produce a difference. The symptom of a comparison
that cannot answer its own question is a suspiciously perfect score, and it
is worth running both configurations to see it. With 20 centres the
comparison becomes real, and coarse coding wins on both targets.

Polynomial features fail on the piecewise target by four orders of magnitude.
A degree 5 polynomial is continuous and has at most four turning points; the
target has two jump discontinuities and a burst of oscillation. No amount of
fitting fixes a representation that cannot express the answer.

One methodological note. The step size here is divided by the largest
eigenvalue of the feature Gram matrix, so that one `alpha` works for all five
constructions. Without that, a polynomial basis on `[-10, 10]` has features of
magnitude `10^5` while tile coding has features of magnitude 1, and a step
size that converges for one diverges for the other. Hand-tuning `alpha` per
method instead would make part of any reported MSE difference a difference in
tuning effort rather than in the representation.

## Parts 3 and 4: semi-gradient TD prediction on a gridworld

### The dataset is generated, not shipped

`make_dataset.py` builds the dataset from
`Gym-Gridworlds/DangerMaze-6x6-v0`, pinned in `requirements.txt`. It collects
3750 transitions under an epsilon-greedy optimal policy with
`epsilon = 0.05`, starting from every one of the 30 states so that the pits
are covered, and stores the exact `v_pi` and `q_pi` of that policy alongside
the data. Generating it rather than shipping a binary keeps the construction
visible and adjustable, and the chapter reproduces from scratch with one
command.

`DangerMaze-6x6` was chosen because its value function has genuine
discontinuities, from `-105` next to `-0.04`. An empty grid produces a smooth
`gamma^distance` surface that all five constructions fit nearly perfectly, so
it cannot discriminate between them. Picking an environment that can
distinguish the methods is part of the experiment design, not a detail.

### What semi-gradient means

The TD objective compares `v(S, w)` with the target `R + gamma v(S', w)`, and
the target depends on `w` as well. The true gradient would contain a
`- gamma grad v(S', w)` term. Semi-gradient methods leave that term out and
treat the target as a constant. That is the whole content of the word.

It is routinely confused with stochastic gradient descent, which is about
using a subset of the data. The two are independent choices and either can be
combined with the other. The distinction has teeth: a semi-gradient update is
not the gradient of any objective function, so the convergence guarantees of
gradient descent do not apply. On-policy linear TD still converges, to the TD
fixed point rather than to the minimum mean squared error (section 9.4), and
off-policy linear TD can diverge outright (section 11.2).

### Results, prediction of `v_pi`

Mean squared error of the learned `v` against the exact `v_pi`, over all 30
states, with 16 features from a 4 by 4 lattice of centres on a 5 by 6 grid:

| construction | features | MSE against true `v_pi` |
|---|---|---|
| polynomial, degree 3 | 10 | 1783.2 |
| RBF | 16 | 640.6 |
| tile coding, 3 tilings | 16 | **419.1** |
| coarse coding, 3 fields | 16 | 735.3 |
| state aggregation | 16 | 534.2 |

Tile coding wins on this task, and coarse coding won the 1D regression in
part 2. That is not a contradiction, and neither result generalises: which
local construction wins depends on how its receptive fields happen to line up
with the discontinuities of the particular value function. The robust
conclusion is the ordering of the polynomial against everything else.

What does generalise is the ordering of the polynomial against everything
else. It is worst by a factor of three, for the same reason as in part 2. A
degree 3 polynomial on a 5 by 6 grid is a smooth surface, and `v_pi` here
drops by a hundred between adjacent tiles. The errors are all large in absolute terms because 16 features cannot
represent 30 values that include jumps of a hundred between neighbours. The
heatmap is worth looking at rather than only the numbers: tile coding and
state aggregation do reproduce some of the pits at roughly the right
magnitude, because their receptive fields are sharp edged and a pit that
falls inside one field can be represented. The polynomial reproduces none of
them, and spreads the error across the whole grid instead. The MSE ranking
and the visual reading agree here, but they do not have to: an approximator
that gets one pit badly wrong and everything else right can score worse than
one that is mediocre everywhere.

### Results, prediction of `q_pi`

Part 4 asks what had to change. Exactly four things, and nothing else:

1. the weights become a `(D, A)` matrix instead of a `(D,)` vector;
2. the prediction for a transition is read from the column of the action
   actually taken;
3. the bootstrap becomes `sum_a' pi(a'|S') q(S', a', w)`, the expectation
   under the policy being evaluated;
4. the gradient is scattered back into that one column only.

With tile coding, the MSE against the exact `q_pi` is 1498.3 over 150 values.
Per entry that is comparable with the `v` result, which is what one would
expect: the same features are being asked to represent five surfaces instead
of one, with five times as many weights.

Note that this is TD *prediction* of `q_pi`, matching part 3. Taking a max in
the bootstrap instead would estimate `q_*`, which is a different quantity and
would not be comparable with the `v_pi` result.

### SL regression against RL TD regression

The similarity is that both minimise a squared error by gradient descent on a
parametric predictor. Three differences do the real work.

The target is not fixed. In supervised regression `y_i` is given; in TD the
target contains the current estimate, so it moves whenever the weights move.
That is what makes the problem non-stationary, and it is why the semi-gradient
distinction above exists at all.

The data is not independent and identically distributed. Consecutive
transitions come from one trajectory and are strongly correlated, and the
distribution over states changes as the policy changes. Chapter 06 is about
one way to put that right.

The data distribution is chosen by the learner. A regression model has no say
in which `x_i` it sees. An agent does, and the states it visits determine
which parts of the value function it can fit, which is the exploration
problem in disguise.

### Continuous actions

The construction used in part 4, one weight column per action, does not
extend to a continuous action space: there is no column to index and no max
to take. The two standard routes are to make the features a function of the
state and the action jointly, `q(s, a, w) = w . phi(s, a)`, so that the action
enters as an input rather than as an index, or to abandon the action value
function and parameterise the policy directly. The second route is chapter 07.

## Figures

| file | content |
|---|---|
| `figures/features_overview.png` | the five constructions at a fixed setting |
| `figures/features_hyperparams.png` | one row per hyperparameter |
| `figures/regression_smooth.png`, `regression_piecewise.png` | part 2 fits and MSE curves |
| `figures/td_v_comparison.png` | true `v_pi` against all five approximations |
| `figures/td_q_heatmaps.png` | true and predicted `q_pi`, one column per action |
| `results/regression_mse.csv`, `results/td_prediction_mse.csv` | the tables above |
