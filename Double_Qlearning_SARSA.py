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
    nA = len(Q)  # number of actions
    probs = np.ones(nA) * eps / nA
    best_action = np.argmax(Q)
    probs[best_action] += (1.0 - eps)
    return probs

# Modified eps_greedy_action function
def eps_greedy_action(Q, s, eps):
    try:
        probs = eps_greedy_probs(Q[s], eps)
        probs /= probs.sum()
        
        
        if np.isnan(probs).any() or probs.sum() == 0:
            raise ValueError(f"Invalid action probabilities: {probs}")
        
        action = np.random.choice(np.arange(len(probs)), p=probs)
        print(f"Action selected: {action}")
        return action
    except Exception as e:
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

# Temporal Difference learning (Double Q-learning, Double SARSA)
def td(env, env_eval, Q1, Q2, gamma, eps, alpha, max_steps, alg):
    be = []  # Bellman error
    exp_ret = []  # Expected return
    tde = np.zeros(max_steps)  # TD errors over all steps
    eps_decay = eps / max_steps  # Epsilon decay
    alpha_decay = alpha / max_steps  # Alpha decay
    tot_steps = 0

    state, _ = env.reset()

    while True:
        action = eps_greedy_action(Q1, state, eps)  # Select action based on Q1
        if action is None:
            raise ValueError("eps_greedy_action did not return a valid action.")

        while True:
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Double Q-Learning: update Q1 based on Q2 and vice versa
            if alg == "D_QL":
                if np.random.rand() < 0.5:
                    next_action = np.argmax(Q1[next_state])
                    td_target = reward + gamma * Q2[next_state][next_action]
                    Q1[state][action] += alpha * (td_target - Q1[state][action])
                else:
                    next_action = np.argmax(Q2[next_state])
                    td_target = reward + gamma * Q1[next_state][next_action]
                    Q2[state][action] += alpha * (td_target - Q2[state][action])

            # Double SARSA: use both Q1 and Q2 in a similar manner
            elif alg == "D_SARSA":
                next_action = eps_greedy_action(Q1, next_state, eps)
                if next_action is None:
                    raise ValueError("eps_greedy_action did not return a valid action.")
                td_target = reward + gamma * Q2[next_state][next_action]
                Q1[state][action] += alpha * (td_target - Q1[state][action])
                action = next_action

            # Double Expected SARSA: average action value comes from both Q1 and Q2
            elif alg == "D_Exp_SARSA":
                action_probs = eps_greedy_probs(Q1[next_state], eps)
                # Compute the expected value using Q2 for the next state-action values
                expected_value_Q2 = np.dot(Q2[next_state], action_probs)
                td_target = reward + gamma * expected_value_Q2
                Q1[state][action] += alpha * (td_target - Q1[state][action])

            # TD update rule
            Q_avg = (Q1 + Q2) / 2
            td_error = td_target - Q_avg[state][action]
            Q_avg[state][action] += alpha * td_error
            tde[tot_steps] = td_error  # Track the TD error
            
            # Move to the next state
            state = next_state
            if done:
                state, _ = env.reset()
                action = eps_greedy_action(Q1, state, eps)

            # Every 100 steps, calculate Bellman error and expected return
            if (tot_steps + 1) % 100 == 0:
                # Use the average of Q1 and Q2 for Bellman error
                pi = np.array([eps_greedy_probs(Q_avg[s], eps) for s in range(n_states)])
                be.append(np.abs(Q_avg - bellman_q(pi, gamma)).mean())
                exp_ret.append(expected_return(env_eval, Q_avg, gamma))

            # Decay epsilon and alpha
            eps = max(eps - eps_decay, 0.01)
            alpha = max(alpha - alpha_decay, 0.001)
            tot_steps += 1
            
            # Break the loop if max_steps is reached
            if tot_steps >= max_steps:
                return (Q1 + Q2) / 2, be, tde, exp_ret



# Smoothing and plotting functions remain the same...


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
algs = ["D_QL", "D_SARSA", "D_Exp_SARSA"]
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

reward_noise_std = 0.0  # re-run with 3.0

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
            # Initialize both Q1 and Q2 for double learning
            Q1 = np.zeros((n_states, n_actions)) + init_value
            Q2 = np.zeros((n_states, n_actions)) + init_value

            # Call the td function with both Q1 and Q2
            Q_avg, be, tde, exp_ret = td(env, env_eval, Q1, Q2, gamma, eps, alpha, max_steps, alg)

            # Store the results
            results_be[i, j, seed] = be
            results_tde[i, j, seed] = tde
            results_exp_ret[i, j, seed] = exp_ret

            print(f"Init value: {init_value}, Algorithm: {alg}, Seed: {seed}")

        # Create labels for the plots
        label = f"$Q_0$: {init_value}, Alg: {alg}"

        # TD Error plot
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

        # Bellman Error plot
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

        # Expected Return plot
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

