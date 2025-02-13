import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize the environment
env = gym.make("Gym-Gridworlds/Penalty-3x3-v0")

# Define the number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize the Reward (R) and Transition Probability (P) matrices
R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))
A = np.zeros((n_states, n_actions))


# Populate R, P, T matrices from the environment dynamics
env.reset()
env.render()

for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated
        print(f"Initial state: {s} Next state: {s_next} Reward: {R[s, a]} Prob: {P[s, a, s_next]} Terminal: {T[s, a]}")

breakpoint
#optimal_policy = np.array([[1, 2, 4], [1, 2, 3], [2, 2, 3]])

# Define the Bellman equation for V (value function)
#
def bellman_v(policy, env, init_value, discount_factor=1.0, theta=0.00001):
    V = np.full(env.observation_space.n, init_value)  # Initialize value function
    #V = np.zeros((env.observation_space.n))  # Initialize value function
    #print(V)


    while True:
        delta = 0  # Initialize convergence threshold

        for s in range(env.observation_space.n):
            v = 0 # Initialize value for state s
            # Compute the sum of policy probabilities for the current state
            sum_policy = 0
            for a in range(env.action_space.n):
                A[s,a] = optimal_policy[s // 3, s % 3] == a  # Policy probability for action a at state s
                

            for a in range(env.action_space.n):
                for next_state in range(env.observation_space.n):
                    reward = R[s, a]
                    # Update value function using the Bellman equation 
                    # The termination term added by multiplying V[next_state] with "(1-T[next_state,a])"
                    v += A[s,a] * P[s, a, next_state] * (reward + (1-T[next_state,a])*discount_factor * V[next_state])

            delta = max(delta, np.abs(v - V[s]))
            V[s] = v

        if delta < theta:
            break

    return np.array(V)



# Define the Bellman equation for Q (action-value function)

def bellman_q(policy, env, init_value, V_value, discount_factor=1.0, theta=0.00001):
    Q = np.full((env.observation_space.n, env.action_space.n), init_value)  # Initialize Q-function
    #Q = np.zeros((env.observation_space.n, env.action_space.n))  # Initialize Q-function
    A = np.zeros((env.observation_space.n, env.action_space.n))  # Initialize policy probability matrix

    while True:
        delta = 0  # Initialize convergence threshold

        for s in range(env.observation_space.n):
            for a in range(env.action_space.n):
                q = 0 # Initialize Q-value for state s and action a

                # Compute the policy probability for each action
                for next_action in range(env.action_space.n):
                    A[s, a] = int(optimal_policy[s // 3, s % 3] == a)  # Convert True/False to 1/0

                # Update Q-function using the Bellman equation
                for next_state in range(env.observation_space.n):
                    reward = R[s, a]
                ##The value of v will be used to compute q
                    q += P[s, a, next_state] * (reward + discount_factor * V_value[next_state])

                delta = max(delta, np.abs(q - Q[s, a]))
                Q[s, a] = q

        if delta < theta:
            break

    return np.array(Q)

# Values of gamma for plotting
gammas = [0.01, 0.5, 0.99]

# Function to annotate heatmap with larger font size
def annotate_heatmap(ax, data, fontsize=16):
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white', fontsize=fontsize)

plt.rcParams.update({'font.size': 14})

# Initial value states for plotting
V_value = np.zeros((env.observation_space.n, env.action_space.n))
for init_value in [-10.0, 0.0, 10.0]:
    fig, axs = plt.subplots(2, len(gammas), figsize=(15, 8))
    fig.suptitle(f"$V_0$: {init_value}", fontsize=16)
    for i, gamma in enumerate(gammas):
        V = bellman_v(optimal_policy, env, init_value, discount_factor=gamma)
        ax = axs[0][i]
        im = ax.imshow(V.reshape(3, 3), cmap='viridis', aspect='auto')
        annotate_heatmap(ax, V.reshape(3, 3), fontsize=14)
        axs[1][i].plot(V)
        ax.set_title(f'$\gamma$ = {gamma}', fontsize=14)

    fig, axs = plt.subplots(n_actions + 1, len(gammas), figsize=(15, 10))
    fig.suptitle(f"$Q_0$: {init_value}", fontsize=16)
    for i, gamma in enumerate(gammas):
        Q = bellman_q(optimal_policy, env, init_value, V, discount_factor=gamma)
        for a in range(n_actions):
            ax = axs[a][i]
            im = ax.imshow(Q[:, a].reshape(3, 3), cmap='viridis', aspect='auto')
            annotate_heatmap(ax, Q[:, a].reshape(3, 3), fontsize=14)
        axs[-1][i].plot(np.max(Q, axis=1))
        axs[0][i].set_title(f'$\gamma$ = {gamma}', fontsize=14)

    plt.show()

