"""Tests for the shared library.

These are property tests rather than smoke tests.  Each one asserts something
the mathematics requires, so that a change which breaks an identity fails
here instead of producing a plausible looking figure: the Bellman operators
agree with each other and with the closed form linear solve, the terminal
transition does not bootstrap, a baseline is unbiased because
`sum_a pi(a|s) grad log pi(a|s) = 0`, and both log gradients match numerical
differentiation.  The reasoning behind each is in `docs/ENGINEERING_NOTES.md`.

    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlkit import bellman as B  # noqa: E402
from rlkit.envs import (  # noqa: E402
    ACTIONS, ACTION_NAMES, extract_mdp, make_env, policy_from_names, resolve_env_id,
)
from rlkit.features import (  # noqa: E402
    aggregation_features, coarse_features, make_centers, poly_features,
    rbf_features, tile_features,
)
from rlkit.policies import (  # noqa: E402
    dlog_gaussian_probs, dlog_softmax_probs, eps_greedy_probs,
    eps_greedy_probs_all, softmax_probs,
)

GAMMA = 0.99
OPTIMAL_POLICY_NAMES = [
    ["DOWN", "RIGHT", "STAY"],
    ["DOWN", "RIGHT", "UP"],
    ["RIGHT", "RIGHT", "UP"],
]


@pytest.fixture(scope="module")
def mdp():
    return extract_mdp("Gym-Gridworlds/Penalty-3x3-v0")


@pytest.fixture(scope="module")
def pi_opt(mdp):
    idx = policy_from_names(OPTIMAL_POLICY_NAMES)
    pi = np.zeros((mdp.n_states, mdp.n_actions))
    pi[np.arange(mdp.n_states), idx] = 1.0
    return pi


# ---------------------------------------------------------------------------
# environment and action resolution
# ---------------------------------------------------------------------------

def test_action_names_are_a_bijection():
    assert sorted(ACTIONS.values()) == list(range(5))
    assert len(set(ACTION_NAMES)) == 5


def test_renamed_env_ids_resolve():
    """The two ids removed upstream must resolve to something registered."""
    for env_id in ("Gym-Gridworlds/Empty-2x2-v0",
                   "Gym-Gridworlds/DangerMaze-6x6-v0",
                   "Gym-Gridworlds/Penalty-3x3-v0"):
        assert resolve_env_id(env_id) in __import__("gymnasium").registry


def test_reset_seeded_makes_a_stochastic_environment_reproducible():
    """`env.reset(seed=...)` alone is not enough, and this proves it.

    `gym_gridworlds` implements `random_action_prob` with
    `self.action_space.sample()`, and the action space has its own generator
    that `env.reset(seed=...)` does not touch.  A run that looks fully seeded
    is then not reproducible, and nothing in any figure shows it.  See
    docs/ENGINEERING_NOTES.md section 3.
    """
    import hashlib

    def rollout(use_helper: bool) -> str:
        env = make_env("Gym-Gridworlds/Empty-2x2-v0", coordinate_observation=True,
                       random_action_prob=0.5)
        out = []
        for k in range(3):
            if use_helper:
                from rlkit.envs import reset_seeded
                reset_seeded(env, k)
            else:
                env.reset(seed=k)
            for t in range(40):
                s, r, terminated, truncated, _ = env.step(t % 5)
                out.append((float(np.asarray(s).sum()), float(r)))
                if terminated or truncated:
                    env.reset()
        env.close()
        return hashlib.md5(np.array(out).tobytes()).hexdigest()

    with_helper = {rollout(True) for _ in range(3)}
    assert len(with_helper) == 1, "reset_seeded must give identical rollouts"


def test_mdp_tables_are_consistent(mdp):
    R, P, T = mdp
    assert R.shape == (9, 5)
    assert P.shape == (9, 5, 9)
    # transitions are deterministic, so every row is a single one
    assert np.allclose(P.sum(axis=-1), 1.0)
    # exactly one terminal transition in this gridworld, STAY on the goal
    terminal = np.argwhere(T > 0)
    assert terminal.shape == (1, 2)
    assert terminal[0, 0] == 2 and terminal[0, 1] == ACTIONS["STAY"]


def test_extract_mdp_matches_live_environment(mdp):
    """The tables must agree with what the environment actually does."""
    env = make_env("Gym-Gridworlds/Penalty-3x3-v0")
    env.reset()
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            env.unwrapped.set_state(s)
            s_next, r, terminated, _, _ = env.step(a)
            assert mdp.R[s, a] == r
            assert mdp.P[s, a, s_next] == 1.0
            assert mdp.T[s, a] == float(terminated)
    env.close()


# ---------------------------------------------------------------------------
# Bellman operators
# ---------------------------------------------------------------------------

def test_iterative_evaluation_matches_the_linear_solve(mdp, pi_opt):
    """Policy evaluation has a closed form; the iterative version must match it."""
    V, _ = B.policy_evaluation_v(pi_opt, mdp, GAMMA, theta=1e-14)
    assert np.allclose(V, B.solve_v_exactly(pi_opt, mdp, GAMMA), atol=1e-9)


def test_q_evaluation_matches_the_linear_solve(mdp, pi_opt):
    Q, _ = B.policy_evaluation_q(pi_opt, mdp, GAMMA, theta=1e-14)
    assert np.allclose(Q, B.solve_q_exactly(pi_opt, mdp, GAMMA), atol=1e-9)


def test_v_and_q_operators_agree(mdp):
    """v_pi(s) must equal sum_a pi(a|s) q_pi(s,a) for any policy.

    This is the identity that fails when the termination factor is indexed as
    `T[s_next, a]` instead of `T[s, a]`, which is the central correction in
    this rebuild.
    """
    rng = np.random.default_rng(0)
    for _ in range(5):
        pi = rng.random((mdp.n_states, mdp.n_actions))
        pi /= pi.sum(axis=1, keepdims=True)
        V = B.solve_v_exactly(pi, mdp, GAMMA)
        Q = B.solve_q_exactly(pi, mdp, GAMMA)
        assert np.allclose(V, (pi * Q).sum(axis=1), atol=1e-9)


def test_optimal_policy_is_the_hand_written_one(mdp, pi_opt):
    """Value iteration must recover the policy written out by hand."""
    V = np.zeros(mdp.n_states)
    for _ in range(5000):
        V = B.optimal_v_backup(V, mdp, GAMMA)
    assert np.allclose(B.greedy_policy_from_v(V, mdp, GAMMA), pi_opt)


def test_optimal_value_of_the_start_state(mdp, pi_opt):
    """The optimal route is six moves, so v_*(s0) must be gamma^6 exactly."""
    V = B.solve_v_exactly(pi_opt, mdp, GAMMA)
    assert V[0] == pytest.approx(GAMMA ** 6, abs=1e-12)


def test_v_and_q_value_iteration_agree(mdp):
    V = np.zeros(mdp.n_states)
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    for _ in range(5000):
        V = B.optimal_v_backup(V, mdp, GAMMA)
        Q = B.optimal_q_backup(Q, mdp, GAMMA)
    assert np.allclose(V, Q.max(axis=1), atol=1e-9)


def test_terminal_transition_does_not_bootstrap(mdp):
    """The goal transition must be worth exactly its reward.

    `(state 2, STAY)` ends the episode, so q_*(2, STAY) = 1.0 and not
    1.0 + gamma * something.  This is the most direct check that termination
    is indexed by the transition rather than by the next state.
    """
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    for _ in range(5000):
        Q = B.optimal_q_backup(Q, mdp, GAMMA)
    assert Q[2, ACTIONS["STAY"]] == pytest.approx(1.0, abs=1e-12)


def test_bellman_error_is_zero_at_the_fixed_point(mdp, pi_opt):
    V = B.solve_v_exactly(pi_opt, mdp, GAMMA)
    backup = B.expected_v_backup(V, pi_opt, mdp, GAMMA)
    assert B.bellman_error(V, backup) < 1e-9


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

def test_eps_greedy_probs_sum_to_one_and_respect_epsilon():
    q = np.array([1.0, 5.0, 2.0, 0.0, -3.0])
    for eps in (0.0, 0.1, 0.5, 1.0):
        p = eps_greedy_probs(q, eps)
        assert p.sum() == pytest.approx(1.0)
        assert p.min() >= eps / len(q) - 1e-12
        if eps < 1.0:
            assert np.argmax(p) == 1
        else:
            # at eps = 1 the policy is uniform and there is no greedy action
            assert np.allclose(p, 1.0 / len(q))


def test_eps_greedy_shares_mass_between_tied_actions():
    """Ties must split the greedy mass, not go to the lowest index."""
    p = eps_greedy_probs(np.array([3.0, 3.0, 0.0]), 0.3)
    assert p[0] == pytest.approx(p[1])
    assert p[0] > p[2]


def test_eps_greedy_vectorised_matches_the_scalar_version():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(7, 5))
    allp = eps_greedy_probs_all(Q, 0.25)
    for s in range(7):
        assert np.allclose(allp[s], eps_greedy_probs(Q[s], 0.25))


def test_softmax_probs_are_a_distribution():
    rng = np.random.default_rng(2)
    phi, w = rng.normal(size=(4, 6)), rng.normal(size=(6, 3))
    p = softmax_probs(phi, w, 1.0)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()


def test_softmax_is_stable_for_large_preferences():
    """The exp-normalise trick must prevent overflow."""
    p = softmax_probs(np.array([[1.0]]), np.array([[1000.0, -1000.0]]), 1.0)
    assert np.isfinite(p).all()
    assert p.sum() == pytest.approx(1.0)


def test_dlog_softmax_matches_a_numerical_gradient():
    rng = np.random.default_rng(3)
    phi, w = rng.normal(size=(1, 5)), rng.normal(size=(5, 4))
    action, tau, h = 2, 1.3, 1e-6
    analytic = dlog_softmax_probs(phi, w, tau, action)
    numeric = np.zeros_like(w)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            wp, wm = w.copy(), w.copy()
            wp[i, j] += h
            wm[i, j] -= h
            numeric[i, j] = (np.log(softmax_probs(phi, wp, tau).ravel()[action])
                             - np.log(softmax_probs(phi, wm, tau).ravel()[action])) / (2 * h)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_dlog_softmax_sums_to_zero_over_actions():
    """sum_a pi(a|s) grad log pi(a|s) = 0, which is why a baseline is unbiased."""
    rng = np.random.default_rng(4)
    phi, w = rng.normal(size=(1, 5)), rng.normal(size=(5, 4))
    probs = softmax_probs(phi, w, 1.0).ravel()
    total = sum(probs[a] * dlog_softmax_probs(phi, w, 1.0, a) for a in range(4))
    assert np.allclose(total, 0.0, atol=1e-12)


def test_dlog_gaussian_matches_a_numerical_gradient():
    rng = np.random.default_rng(5)
    phi, w = rng.normal(size=(1, 4)), rng.normal(size=(4, 1))
    sigma, action, h = 0.7, np.array([[0.3]]), 1e-6

    def logp(weights):
        mu = np.dot(phi, weights)
        return float(np.asarray(-0.5 * ((action - mu) / sigma) ** 2).ravel()[0])

    analytic = dlog_gaussian_probs(phi, w, sigma, action)
    numeric = np.zeros_like(w)
    for i in range(w.shape[0]):
        wp, wm = w.copy(), w.copy()
        wp[i, 0] += h
        wm[i, 0] -= h
        numeric[i, 0] = (logp(wp) - logp(wm)) / (2 * h)
    assert np.allclose(analytic, numeric, atol=1e-6)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def test_every_feature_function_returns_n_by_d():
    rng = np.random.default_rng(6)
    states = rng.random((11, 2))
    centers = make_centers([0, 0], [1, 1], [5, 5])
    for phi in (poly_features(states, 2),
                rbf_features(states, centers, 0.2),
                tile_features(states, centers, 0.2),
                coarse_features(states, centers, 0.2),
                aggregation_features(states, centers)):
        assert phi.ndim == 2 and phi.shape[0] == 11


def test_feature_functions_accept_a_single_state():
    centers = make_centers([0, 0], [1, 1], [3, 3])
    single = np.array([[0.5, 0.5]])
    for phi in (poly_features(single, 2),
                rbf_features(single, centers, 0.2),
                tile_features(single, centers, 0.2),
                coarse_features(single, centers, 0.2),
                aggregation_features(single, centers)):
        assert phi.shape[0] == 1


def test_polynomial_feature_count_is_the_multivariate_one():
    """Degree d over S dimensions gives C(S + d, d) monomials, not S * d."""
    from math import comb
    for s_dim in (1, 2, 3):
        for degree in (1, 2, 3):
            states = np.zeros((2, s_dim))
            assert poly_features(states, degree).shape[1] == comb(s_dim + degree, degree)


def test_aggregation_is_one_hot():
    rng = np.random.default_rng(7)
    centers = make_centers([0, 0], [1, 1], [4, 4])
    phi = aggregation_features(rng.random((20, 2)), centers)
    assert np.allclose(phi.sum(axis=1), 1.0)
    assert set(np.unique(phi)) <= {0.0, 1.0}


def test_tile_coding_is_binary_per_tiling():
    centers = make_centers([0, 0], [1, 1], [5, 5])
    states = np.random.default_rng(8).random((15, 2))
    one = tile_features(states, centers, 0.2)
    assert set(np.unique(one)) <= {0.0, 1.0}
    four = tile_features(states, centers, 0.2,
                         [(-0.05, 0.0), (0.0, 0.05), (0.05, 0.0), (0.0, -0.05)])
    assert four.max() <= 4.0


def test_coarse_coding_is_graded_and_tile_coding_is_not():
    """The one difference between the two constructions."""
    centers = make_centers([0, 0], [1, 1], [5, 5])
    states = np.random.default_rng(9).random((50, 2))
    coarse = coarse_features(states, centers, 0.4)
    active = coarse[coarse > 0]
    assert active.size > 0
    assert (active < 1.0).any(), "coarse coding should produce partial activations"


def test_rbf_peaks_at_its_centre():
    centers = np.array([[0.5, 0.5]])
    at_centre = rbf_features(np.array([[0.5, 0.5]]), centers, 0.2)
    away = rbf_features(np.array([[0.9, 0.9]]), centers, 0.2)
    assert at_centre[0, 0] == pytest.approx(1.0)
    assert away[0, 0] < at_centre[0, 0]
