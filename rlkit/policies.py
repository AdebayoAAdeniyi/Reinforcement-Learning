"""Policies: epsilon-greedy, softmax and Gaussian, with their log gradients.

These are shared by the Monte Carlo, temporal difference, fitted Q-iteration,
policy gradient and actor-critic chapters, so that a policy means the same
thing in all of them and the log gradients are tested in one place rather
than reimplemented per experiment.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "eps_greedy_probs",
    "eps_greedy_probs_all",
    "eps_greedy_action",
    "softmax_probs",
    "softmax_action",
    "dlog_softmax_probs",
    "gaussian_action",
    "dlog_gaussian_probs",
]


# ----------------------------------------------------------------------------
# epsilon-greedy
# ----------------------------------------------------------------------------

def eps_greedy_probs(q: np.ndarray, eps: float) -> np.ndarray:
    """Epsilon-greedy action probabilities for one state.  `q` has shape (A,).

    Probability mass eps is spread uniformly over all A actions and the
    remaining 1 - eps goes to the greedy action.  Ties for the greedy action
    share the 1 - eps equally, which keeps the distribution symmetric in
    states where several actions are genuinely equally good.  `np.argmax`
    would hand the whole 1 - eps to the lowest index instead, which biases
    exploration in precisely those states; several actions in these
    gridworlds bump into a wall and are exactly equivalent.
    """
    q = np.asarray(q, dtype=float)
    n = q.shape[-1]
    probs = np.full(n, eps / n)
    best = q == q.max()
    probs[best] += (1.0 - eps) / best.sum()
    return probs


def eps_greedy_probs_all(Q: np.ndarray, eps: float) -> np.ndarray:
    """Vectorised `eps_greedy_probs` over every state.  `Q` has shape (S, A)."""
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[-1]
    probs = np.full(Q.shape, eps / n)
    best = Q == Q.max(axis=-1, keepdims=True)
    probs += best * (1.0 - eps) / best.sum(axis=-1, keepdims=True)
    return probs


def eps_greedy_action(q: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    """Sample an epsilon-greedy action for one state."""
    return int(rng.choice(len(q), p=eps_greedy_probs(q, eps)))


# ----------------------------------------------------------------------------
# softmax, for discrete action policy gradient
# ----------------------------------------------------------------------------

def softmax_probs(phi: np.ndarray, weights: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax over linear preferences h(s, a) = phi(s) . w[:, a].

    `phi` has shape (N, D), `weights` has shape (D, A), the result (N, A).
    Uses the exp-normalise trick, https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Sutton and Barto call this soft-max in action preferences, section 13.2.
    """
    h = np.dot(phi, weights) / max(temperature, 1e-12)
    h = h - h.max(axis=-1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_action(phi: np.ndarray, weights: np.ndarray, temperature, rng) -> int:
    """Sample one action from the softmax policy."""
    probs = softmax_probs(phi, weights, temperature).ravel()
    return int(rng.choice(probs.shape[0], p=probs))


def dlog_softmax_probs(
    phi: np.ndarray, weights: np.ndarray, temperature: float, action: int
) -> np.ndarray:
    """Gradient of log pi(a | s) with respect to the weights.

    For soft-max in linear action preferences divided by a temperature tau,

        d log pi(a|s) / d w[:, i] = phi(s) ( 1[i = a] - pi(i|s) ) / tau

    which is Sutton and Barto equation 13.9 with the eligibility vector
    x(s, a) = phi(s) e_a.  The `1 / tau` factor is easy to drop: with tau = 1
    that is invisible, and with any other temperature it silently rescales the
    learning rate.  `test_dlog_softmax_matches_a_numerical_gradient` checks
    the whole expression at tau = 1.3 for that reason.

    `phi` has shape (1, D) or (D,), the result has shape (D, A).
    """
    probs = softmax_probs(phi, weights, temperature).ravel()      # (A,)
    phi = np.asarray(phi, dtype=float).ravel()                    # (D,)
    onehot = np.zeros_like(probs)
    onehot[action] = 1.0
    return np.outer(phi, onehot - probs) / max(temperature, 1e-12)


# ----------------------------------------------------------------------------
# Gaussian, for continuous action policy gradient
# ----------------------------------------------------------------------------

def gaussian_action(phi: np.ndarray, weights: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Sample from N(phi . w, sigma^2).

    Note the parameterisation: `sigma` is the standard deviation throughout,
    here and in `dlog_gaussian_probs`, so the variance is `sigma ** 2`.
    `np.random.normal` takes the standard deviation, and the log gradient
    divides by the variance; passing `sigma ** 2` to the sampler while still
    dividing by `sigma ** 2` in the gradient leaves it off by a factor of
    sigma squared, which acts as an uncontrolled learning rate schedule
    whenever sigma is annealed.
    """
    mu = np.dot(phi, weights)
    return rng.normal(mu, sigma)


def dlog_gaussian_probs(
    phi: np.ndarray, weights: np.ndarray, sigma: float, action: np.ndarray
) -> np.ndarray:
    """Gradient of log pi(a | s) with respect to the mean weights.

    For a Gaussian policy with fixed standard deviation sigma,

        d log pi(a|s) / d w = (a - mu(s)) / sigma^2  *  phi(s)

    Sutton and Barto, section 13.7, equation 13.19.  Only the mean is learned
    here; sigma follows a fixed decay schedule.
    """
    mu = np.dot(phi, weights)                                     # (1, A)
    phi = np.asarray(phi, dtype=float).reshape(-1, 1)             # (D, 1)
    return phi * np.asarray((action - mu), dtype=float).reshape(1, -1) / (sigma ** 2)
