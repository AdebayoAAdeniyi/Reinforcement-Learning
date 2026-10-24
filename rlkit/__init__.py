"""rlkit: the shared pieces behind every chapter of this repository.

Every definition that more than one chapter depends on lives here and is
tested once, so that results from different chapters are strictly comparable:
the Bellman error means the same thing in chapters 02, 03 and 04, the
epsilon-greedy policy breaks ties the same way everywhere, and a change to a
feature construction reaches every experiment that uses it.

Modules
-------
`envs`        tabular MDP extraction, version robust action names and env ids
`bellman`     Bellman operators, Bellman error, policy evaluation, greedy policies
`policies`    epsilon-greedy, softmax and Gaussian policies and their log gradients
`features`    polynomial, RBF, tile, coarse and aggregation features
`evaluation`  expected return by rollout, seeding helpers
`plotting`    smoothing, confidence bands, figure saving
"""

from . import bellman, envs, evaluation, features, plotting, policies  # noqa: F401

__all__ = ["bellman", "envs", "evaluation", "features", "plotting", "policies"]
__version__ = "1.0.0"
