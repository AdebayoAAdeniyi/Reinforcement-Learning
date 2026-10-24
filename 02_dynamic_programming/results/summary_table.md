# Policy evaluation steps to convergence

One step = one Bellman backup over the whole state space. gamma = 0.99, theta = 1e-10.

### Learning `v_*`

| algorithm | -100 | -10 | -5 | 0 | 5 | 10 | 100 | mean +/- std |
|---|---|---|---|---|---|---|---|---|
| VI | 8 | 8 | 8 | 8 | 168 | 237 | 466 | **129.0 +/- 162.7** |
| PI | 3376 | 3465 | 3467 | 3470 | 3472 | 3474 | 3500 | **3460.6 +/- 36.2** |
| GPI | 19 | 37 | 37 | 37 | 72 | 217 | 487 | **129.4 +/- 158.9** |

### Learning `q_*`

| algorithm | -100 | -10 | -5 | 0 | 5 | 10 | 100 | mean +/- std |
|---|---|---|---|---|---|---|---|---|
| VI | 9 | 9 | 9 | 9 | 169 | 238 | 467 | **130.0 +/- 162.7** |
| PI | 3603 | 3691 | 3694 | 3696 | 3698 | 3700 | 3726 | **3686.9 +/- 35.9** |
| GPI | 19 | 37 | 37 | 37 | 142 | 242 | 492 | **143.7 +/- 160.7** |

### Sensitivity of the comparison to the tolerance

Mean policy evaluation steps over the seven initialisations, learning `v_*`.

| theta | VI | PI | GPI |
|---|---|---|---|
| 0.0001 | 129.0 | 1517.4 | 129.4 |
| 1e-06 | 129.0 | 2165.1 | 129.4 |
| 1e-08 | 129.0 | 2813.0 | 129.4 |
| 1e-10 | 129.0 | 3460.6 | 129.4 |
