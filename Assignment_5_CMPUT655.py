import gymnasium
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize reward, transition probability, and terminal state arrays
R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

# Populate R, P, and T based on the environment dynamics
env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.unwrapped.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

P = P * (1.0 - T[..., None])  # Set next state probability to 0 for terminal transitions

# Bellman Q-value computation
def bellman_q(pi, gamma, max_iter=1000):
    delta = np.inf
    iter = 0
    Q = np.zeros((n_states, n_actions))
    be = np.zeros((max_iter))
    while delta > 1e-5 and iter < max_iter:
        Q_new = R + (np.dot(P, gamma * (Q * pi)).sum(-1))
        delta = np.abs(Q_new - Q).sum()
        be[iter] = delta
        Q = Q_new
        iter += 1
    return Q

# Epsilon-greedy policy probabilities

def eps_greedy_probs(Q, eps):
    """
    Calculate epsilon-greedy action probabilities.
    """
    nA = len(Q)  # number of actions
    probs = np.ones(nA) * eps / nA
    best_action = np.argmax(Q)
    probs[best_action] += (1.0 - eps)
    return probs

# Modified eps_greedy_action function
def eps_greedy_action(Q, s, eps):
    """
    Select an action using epsilon-greedy policy based on state s.
    """
    try:
        # Compute epsilon-greedy probabilities
        probs = eps_greedy_probs(Q[s], eps)
        
        # Check if probabilities sum to 1 (fix floating-point precision issues)
        probs /= probs.sum()

        
        # Select an action based on the probabilities
        if np.isnan(probs).any() or probs.sum() == 0:
            raise ValueError(f"Invalid action probabilities: {probs}")
        
        action = np.random.choice(np.arange(len(probs)), p=probs)
        
        # Debugging: print the selected action
        print(f"Action selected: {action}")
        
        return action
    
    except Exception as e:
        # If an error occurs, print detailed information
        print(f"Error in eps_greedy_action: {e}")
        print(f"State: {s}, Q-values: {Q[s]}")
        return None


# Compute expected return for a given policy
def expected_return(env, Q, gamma, episodes=10):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            a = eps_greedy_action(Q, s, 0.0)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()

# Temporal Difference learning (SARSA, Q-learning, Expected SARSA)

def td(env, env_eval, Q, gamma, eps, alpha, max_steps, alg):
    be = []
    exp_ret = []
    tde = np.zeros(max_steps)  # Store TD errors over all steps
    eps_decay = eps / max_steps  # Epsilon decay
    alpha_decay = alpha / max_steps  # Alpha decay
    tot_steps = 0

    # Initialize environment
    state, _ = env.reset()

    # Begin the loop until max_steps is reached
    while True:  # Outer loop for steps
        # Choose an action using the epsilon-greedy strategy
        action = eps_greedy_action(Q, state, eps)

        if action is None:
            raise ValueError("eps_greedy_action did not return a valid action.")

        # Inner loop for episodes
        while True:
            # Take the action and observe the next state and reward
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Calculate TD target based on the selected algorithm
            if alg == "QL":
                td_target = reward + gamma * np.max(Q[next_state])
            elif alg == "SARSA":
                next_action = eps_greedy_action(Q, next_state, eps)
                td_target = reward + gamma * Q[next_state][next_action]
            elif alg == "Exp_SARSA":
                action_probs = eps_greedy_probs(Q[next_state], eps)
                td_target = reward + gamma * np.dot(Q[next_state], action_probs)
            else:
                raise ValueError("Unknown algorithm type")

            # TD update rule
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            tde[tot_steps] = td_error  # Track the TD error
            
            # Move to the next state
            state = next_state
            
            # Update action if using SARSA
            if alg == "SARSA":
                action = next_action
            elif alg == "QL":
                action = eps_greedy_action(Q, state, eps)

            # Record Bellman error and expected return at intervals
            if (tot_steps + 1) % 100 == 0:
                pi = np.array([eps_greedy_probs(Q[s], eps) for s in range(n_states)])
                be.append(np.abs(Q - bellman_q(pi, gamma)).mean())
                exp_ret.append(expected_return(env_eval, Q, gamma))

            # Reset if done
            if done:
                state, _ = env.reset()  # Reset environment
                action = eps_greedy_action(Q, state, eps)  # Choose new action
            
            # Decay epsilon and alpha
            eps = max(eps - eps_decay, 0.01)
            alpha = max(alpha - alpha_decay, 0.001)
            
            tot_steps += 1
            
            # Check for termination condition
            if tot_steps >= max_steps:
                return Q, be, tde, exp_ret

    # The outer loop will terminate through the break when tot_steps >= max_steps






# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

gamma = 0.99
alpha = 0.1
eps = 1.0
max_steps = 10000
horizon = 10

init_values = [-10, 0.0, 10]
algs = ["QL", "SARSA", "Exp_SARSA"]
seeds = np.arange(10)

results_be = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps // 100,
))
results_tde = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps,
))
results_exp_ret = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps // 100,
))

fig, axs = plt.subplots(1, 3)
plt.ion()
plt.show()

reward_noise_std = 3.0  # re-run with 3.0

for ax in axs:
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")

env = gymnasium.make(
    "Gym-Gridworlds/Penalty-3x3-v0",
    max_episode_steps=horizon,
    reward_noise_std=reward_noise_std,
)

env_eval = gymnasium.make(
    "Gym-Gridworlds/Penalty-3x3-v0",
    max_episode_steps=horizon,
)

for i, init_value in enumerate(init_values):
    for j, alg in enumerate(algs):
        for seed in seeds:
            np.random.seed(seed)
            Q = np.zeros((n_states, n_actions)) + init_value
            Q, be, tde, exp_ret = td(env, env_eval, Q, gamma, eps, alpha, max_steps, alg)
            results_be[i, j, seed] = be
            results_tde[i, j, seed] = tde
            results_exp_ret[i, j, seed] = exp_ret
            print(i, j, seed)
        label = f"$Q_0$: {init_value}, Alg: {alg}"
        axs[0].set_title("TD Error")
        error_shade_plot(
            axs[0],
            results_tde[i, j],
            stepsize=1,
            smoothing_window=20,
            label=label,
        )
        axs[0].legend()
        axs[0].set_ylim([0, 5])
        axs[1].set_title("Bellman Error")
        error_shade_plot(
            axs[1],
            results_be[i, j],
            stepsize=100,
            smoothing_window=20,
            label=label,
        )
        axs[1].legend()
        axs[1].set_ylim([0, 50])
        axs[2].set_title("Expected Return")
        error_shade_plot(
            axs[2],
            results_exp_ret[i, j],
            stepsize=100,
            smoothing_window=20,
            label=label,
        )
        axs[2].legend()
        axs[2].set_ylim([-5, 1])
        plt.draw()
        plt.pause(0.001)

plt.ioff()
plt.show()
