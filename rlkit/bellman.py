"""Bellman operators and Bellman errors for finite MDPs.

Every algorithm in this repository that needs a ground truth value function
calls one of these.  They are written once, here, so that the definition of
the Bellman error is identical in the dynamic programming chapter, the Monte
Carlo chapter and the temporal difference chapter, and the numbers reported in
different chapters can honestly be compared.

Reference: Sutton and Barto, *Reinforcement Learning: An Introduction*,
2nd edition, equations 3.14 (v_pi), 3.13 (q_pi), 4.5 (iterative policy
evaluation) and section 4.4 (value iteration).

Notation and the termination factor
-----------------------------------
For a deterministic gridworld the four argument dynamics p(s', r | s, a)
factor into a reward table R[s, a] and a transition table P[s, a, s'].
The episode ends on the transition *leaving* (s, a), which the environment
reports as `terminated` and which is stored in T[s, a].  The bootstrapped term
must therefore be switched off with `(1 - T[s, a])`:

    v(s)   = sum_a pi(a|s) ( R[s,a] + (1 - T[s,a]) gamma sum_s' P[s,a,s'] v(s') )
    q(s,a) =       R[s,a] + (1 - T[s,a]) gamma sum_s' P[s,a,s'] sum_a' pi(a'|s') q(s',a')

`(1 - T[s_next, a])` asks a different question, namely whether the *next*
state would terminate under the *same* action index.  It is nonzero in states
the agent is merely passing through, and it moves the fixed point without
producing anything that looks wrong in a figure, which is why two tests pin
the correct form down.  See docs/ENGINEERING_NOTES.md section 2.
"""

from __future__ import annotations

import numpy as np

from .envs import MDP

__all__ = [
    "expected_v_backup",
    "expected_q_backup",
    "optimal_v_backup",
    "optimal_q_backup",
    "bellman_error",
    "policy_evaluation_v",
    "policy_evaluation_q",
    "greedy_policy_from_q",
    "greedy_policy_from_v",
    "q_from_v",
    "solve_v_exactly",
    "solve_q_exactly",
]


# ----------------------------------------------------------------------------
# one application of a Bellman operator
# ----------------------------------------------------------------------------

def expected_v_backup(V: np.ndarray, pi: np.ndarray, mdp: MDP, gamma: float) -> np.ndarray:
    """One sweep of the Bellman expectation operator for v_pi.  Returns (S,)."""
    R, P, T = mdp
    q = R + (1.0 - T) * gamma * (P @ V)          # (S, A)
    return (pi * q).sum(axis=1)


def expected_q_backup(Q: np.ndarray, pi: np.ndarray, mdp: MDP, gamma: float) -> np.ndarray:
    """One sweep of the Bellman expectation operator for q_pi.  Returns (S, A)."""
    R, P, T = mdp
    V_next = (pi * Q).sum(axis=1)                 # (S,)
    return R + (1.0 - T) * gamma * (P @ V_next)


def optimal_v_backup(V: np.ndarray, mdp: MDP, gamma: float) -> np.ndarray:
    """One sweep of the Bellman optimality operator for v_*.  Returns (S,)."""
    R, P, T = mdp
    return (R + (1.0 - T) * gamma * (P @ V)).max(axis=1)


def optimal_q_backup(Q: np.ndarray, mdp: MDP, gamma: float) -> np.ndarray:
    """One sweep of the Bellman optimality operator for q_*.  Returns (S, A)."""
    R, P, T = mdp
    return R + (1.0 - T) * gamma * (P @ Q.max(axis=1))


# ----------------------------------------------------------------------------
# Bellman error
# ----------------------------------------------------------------------------

def bellman_error(value: np.ndarray, backup: np.ndarray) -> float:
    """Total absolute Bellman error, the quantity the assignments ask to log.

    This is the L1 norm of the residual, `sum |T v - v|`, not the max norm and
    not the mean.  One definition is used everywhere in this repository so
    that curves from different chapters sit on the same axis.
    """
    return float(np.abs(np.asarray(backup) - np.asarray(value)).sum())


# ----------------------------------------------------------------------------
# iterative policy evaluation
# ----------------------------------------------------------------------------

def policy_evaluation_v(
    pi: np.ndarray,
    mdp: MDP,
    gamma: float,
    init_value: float = 0.0,
    theta: float = 1e-10,
    max_sweeps: int = 100_000,
) -> tuple[np.ndarray, list[float]]:
    """Iterative policy evaluation for v_pi (Sutton and Barto, section 4.1).

    Returns the converged V and the Bellman error after every sweep.  The
    sweep is synchronous: the whole new V is computed from the whole old V.
    Gauss-Seidel style in-place updates converge faster but are a different
    algorithm, and mixing the two between algorithms makes an iteration count
    comparison meaningless.  All three algorithms in chapter 02 use this one.
    """
    V = np.full(mdp.n_states, float(init_value))
    errors: list[float] = []
    for _ in range(max_sweeps):
        V_new = expected_v_backup(V, pi, mdp, gamma)
        err = bellman_error(V, V_new)
        errors.append(err)
        V = V_new
        if err < theta:
            break
    return V, errors


def policy_evaluation_q(
    pi: np.ndarray,
    mdp: MDP,
    gamma: float,
    init_value: float = 0.0,
    theta: float = 1e-10,
    max_sweeps: int = 100_000,
) -> tuple[np.ndarray, list[float]]:
    """Iterative policy evaluation for q_pi.  Same contract as the V version."""
    Q = np.full((mdp.n_states, mdp.n_actions), float(init_value))
    errors: list[float] = []
    for _ in range(max_sweeps):
        Q_new = expected_q_backup(Q, pi, mdp, gamma)
        err = bellman_error(Q, Q_new)
        errors.append(err)
        Q = Q_new
        if err < theta:
            break
    return Q, errors


# ----------------------------------------------------------------------------
# greedy policies
# ----------------------------------------------------------------------------

def q_from_v(V: np.ndarray, mdp: MDP, gamma: float) -> np.ndarray:
    """q(s, a) implied by a state value function, needed for improvement."""
    R, P, T = mdp
    return R + (1.0 - T) * gamma * (P @ V)


def greedy_policy_from_q(Q: np.ndarray) -> np.ndarray:
    """Deterministic greedy policy as a one hot (S, A) matrix.

    Ties are broken by the lowest action index, deterministically, so that
    `assert np.allclose(pi_learned, pi_opt)` is a meaningful check rather than
    a coin flip.  Ties do occur here: in `Penalty-3x3-v0` several actions bump
    into a wall and give identical values.
    """
    pi = np.zeros_like(Q)
    pi[np.arange(Q.shape[0]), Q.argmax(axis=1)] = 1.0
    return pi


def greedy_policy_from_v(V: np.ndarray, mdp: MDP, gamma: float) -> np.ndarray:
    """Deterministic greedy policy with respect to the q implied by V."""
    return greedy_policy_from_q(q_from_v(V, mdp, gamma))


# ----------------------------------------------------------------------------
# closed form reference
# ----------------------------------------------------------------------------

def solve_v_exactly(pi: np.ndarray, mdp: MDP, gamma: float) -> np.ndarray:
    """Solve v_pi = R_pi + gamma P_pi v_pi as a linear system.

    Policy evaluation does not have to be iterative.  For a finite MDP the
    Bellman expectation equation is |S| linear equations in |S| unknowns and
    can be solved directly in O(|S|^3).  This is used in the tests as an
    independent check on the iterative routines, and it is the answer to the
    Assignment 3 question "policy evaluation can be solved in a way that is
    not iterative, describe it".
    """
    R, P, T = mdp
    R_pi = (pi * R).sum(axis=1)                              # (S,)
    P_pi = (pi * (1.0 - T))[:, :, None] * P                  # (S, A, S)
    P_pi = P_pi.sum(axis=1)                                  # (S, S)
    return np.linalg.solve(np.eye(mdp.n_states) - gamma * P_pi, R_pi)


def solve_q_exactly(pi: np.ndarray, mdp: MDP, gamma: float) -> np.ndarray:
    """Solve q_pi = R + gamma (1 - T) P Pi q_pi as a linear system.

    Flattened over (s, a), this is |S||A| equations in |S||A| unknowns with

        M[(s,a), (s',a')] = (1 - T[s,a]) P[s,a,s'] pi(a'|s')

    Used wherever q_pi is needed as a reference inside a learning loop.  It
    costs one O((|S||A|)^3) solve instead of hundreds of iterative sweeps, and
    unlike the iterative version its accuracy does not depend on a tolerance.
    """
    R, P, T = mdp
    S, A = mdp.n_states, mdp.n_actions
    M = (1.0 - T)[:, :, None, None] * P[:, :, :, None] * pi[None, None, :, :]
    M = M.reshape(S * A, S * A)
    return np.linalg.solve(np.eye(S * A) - gamma * M, R.reshape(-1)).reshape(S, A)
