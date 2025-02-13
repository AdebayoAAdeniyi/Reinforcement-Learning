import time
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

# Define the Bellman equation for Q (action-value function)
def policy_evaluation(Q, policy, discount_factor=0.99):
    new_Q = np.copy(Q)
    for s in range(n_states):
        for a in range(n_actions):
            q = 0
            for s_next in range(n_states):
                q += P[s, a, s_next] * (R[s, a] + (1 - T[s_next, a]) * discount_factor * np.max(Q[s_next]))
            new_Q[s, a] = q
    return new_Q

# Define Policy Improvement
def policy_improvement(Q, policy):
    policy_stable = True
    for s in range(n_states):
        old_action = policy[s // 3, s % 3]
        # Pick the action that maximizes the Q-value
        new_action = np.argmax(Q[s])
        policy[s // 3, s % 3] = new_action
        if old_action != new_action:
            policy_stable = False
    return policy, policy_stable

# Policy Iteration with Q-values
def policy_iteration(discount_factor=0.99, theta=1e-6, Q_init=None):
    optimal_policy = np.array([[1, 2, 4], [1, 2, 3], [2, 2, 3]])  # Given optimal policy
    policy = np.random.randint(0, n_actions, size=(3, 3))  # Random initial policy
    Q = np.copy(Q_init) if Q_init is not None else np.zeros((n_states, n_actions))
    policy_stable = False
    policy_evaluations = 0
    bellman_errors = []
    iteration_number = 0

    while not policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            Q_prev = np.copy(Q)
            Q = policy_evaluation(Q, policy, discount_factor)
            delta = np.max(np.abs(Q - Q_prev))
            bellman_errors.append(delta)
            if delta < theta or iteration_number >= 700:
                break
            policy_evaluations += 1
            iteration_number += 1

        # Policy Improvement
        policy, policy_stable = policy_improvement(Q, policy)
    
    # Check if the final policy matches the optimal policy
    policy_matches_optimal = np.allclose(policy, optimal_policy)
    return policy, policy_evaluations, Q, bellman_errors, policy_matches_optimal



# Generalized Policy Iteration with custom initial V
def generalized_policy_iteration(discount_factor=0.99, theta=1e-6, Q_init=None, eval_steps=5):
    optimal_policy = np.array([[1, 2, 4], [1, 2, 3], [2, 2, 3]])  # Given optimal policy
    policy = np.random.randint(0, n_actions, size=(3, 3))  # Random initial policy
    Q = np.copy(Q_init) if Q_init is not None else np.zeros((n_states, n_actions))  # Initialize Q-values
    policy_stable = False
    policy_evaluations = 0
    bellman_errors = []

    while not policy_stable:
        # Policy Evaluation: Update Q(s, a) with a fixed number of steps (eval_steps)
        for _ in range(eval_steps):
            delta = 0
            Q_prev = np.copy(Q)  # Keep track of previous Q-values
            for s in range(n_states):
                for a in range(n_actions):
                    # Compute Q(s, a) using Bellman expectation equation for Q-values
                    Q[s, a] = sum(P[s, a, s_next] * (R[s, a] + (1 - T[s_next, a]) * discount_factor * np.max(Q_prev[s_next])) for s_next in range(n_states))
            delta = np.max(np.abs(Q - Q_prev))  # Bellman error
            bellman_errors.append(delta)
            policy_evaluations += 1

        # Policy Improvement: Greedily update the policy based on Q-values
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s // 3, s % 3]
            new_action = np.argmax(Q[s])  # Select the action that maximizes Q(s, a)
            policy[s // 3, s % 3] = new_action
            if old_action != new_action:
                policy_stable = False

    # Check if the final policy matches the optimal policy
    policy_matches_optimal = np.allclose(policy, optimal_policy)
    return policy, policy_evaluations, Q, bellman_errors, policy_matches_optimal


def generalized_policy_iteration(discount_factor=0.99, theta=1e-6, Q_init=None, eval_steps=5, max_iterations=700):
    optimal_policy = np.array([[1, 2, 4], [1, 2, 3], [2, 2, 3]])  # Given optimal policy
    policy = np.random.randint(0, n_actions, size=(3, 3))  # Random initial policy
    Q = np.copy(Q_init) if Q_init is not None else np.zeros((n_states, n_actions))  # Initialize Q-values
    policy_stable = False
    policy_evaluations = 0
    bellman_errors = []
    iteration_number = 0  # Added iteration counter

    while not policy_stable:
        # Stop if the number of iterations exceeds max_iterations
        if iteration_number >= max_iterations:
            break
        
        # Policy Evaluation: Update Q(s, a) with a fixed number of steps (eval_steps)
        for _ in range(eval_steps):
            delta = 0
            Q_prev = np.copy(Q)  # Keep track of previous Q-values
            for s in range(n_states):
                for a in range(n_actions):
                    # Compute Q(s, a) using Bellman expectation equation for Q-values
                    Q[s, a] = sum(P[s, a, s_next] * (R[s, a] + (1 - T[s_next, a]) * discount_factor * np.max(Q_prev[s_next])) for s_next in range(n_states))
            delta = np.max(np.abs(Q - Q_prev))  # Bellman error
            bellman_errors.append(delta)
            policy_evaluations += 1

        # Policy Improvement: Greedily update the policy based on Q-values
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s // 3, s % 3]
            new_action = np.argmax(Q[s])  # Select the action that maximizes Q(s, a)
            policy[s // 3, s % 3] = new_action
            if old_action != new_action:
                policy_stable = False
        
        iteration_number += 1  # Increment iteration count

    # Extract the final policy from the Q-function
    for s in range(n_states):
        best_action = np.argmax(Q[s])
        policy[s // 3, s % 3] = best_action

    # Check if the final policy matches the optimal policy
    policy_matches_optimal = np.allclose(policy, optimal_policy)
    
    return policy, iteration_number, Q, bellman_errors, policy_matches_optimal



# Value Iteration with Q-values
def value_iteration(discount_factor=0.99, theta=1e-6, Q_init=None):
    optimal_policy = np.array([[1, 2, 4], [1, 2, 3], [2, 2, 3]])  # Given optimal policy
    Q = np.copy(Q_init) if Q_init is not None else np.zeros((n_states, n_actions))
    policy = np.zeros((3, 3), dtype=int)
    bellman_errors = []
    iteration_number = 0

    while True:
        delta = 0
        Q_prev = np.copy(Q)
        for s in range(n_states):
            for a in range(n_actions):
                q = 0
                for s_next in range(n_states):
                    q += P[s, a, s_next] * (R[s, a] + (1 - T[s_next, a]) * discount_factor * np.max(Q[s_next]))
                Q[s, a] = q
            bellman_errors.append(np.max(np.abs(Q - Q_prev)))
        delta = np.max(np.abs(Q - Q_prev))
        if delta < theta or iteration_number >= 700:
            break
        iteration_number += 1

    # Extract the policy from the final Q-function
    for s in range(n_states):
        best_action = np.argmax(Q[s])
        policy[s // 3, s % 3] = best_action

    # Check if the final policy matches the optimal policy
    policy_matches_optimal = np.allclose(policy, optimal_policy)
    return policy, iteration_number, Q, bellman_errors, policy_matches_optimal



# Log and plot results
def log_and_plot_results_q():
    init_values = [-100.0, -10.0, -5.0, 0.0, 5.0, 10.0, 100.0]
    tot_iter_table = np.zeros((3, len(init_values)))  # Rows: VI, PI, GPI; Columns: different initial values
    times_table = np.zeros((3, len(init_values)))  # Rows: VI, PI, GPI; Columns: different initial values
    matches_optimal_table = np.zeros((3, len(init_values)), dtype=bool)  # Rows: VI, PI, GPI; Columns: different initial values

    fig, axs = plt.subplots(3, 7, figsize=(20, 15))  # Create subplots for Bellman errors
    heatmap_fig, heatmap_axs = plt.subplots(3, 7, figsize=(20, 15))  # Create subplots for heatmaps

    for i, init_value in enumerate(init_values):
        # Initialize the Q-values
        Q_init = np.full((n_states, n_actions), init_value)

        # Value Iteration
        start_time = time.time()
        pi_vi, tot_iter_vi, Q_vi, be_vi, matches_optimal_vi = value_iteration(discount_factor=0.99, Q_init=Q_init)
        tot_iter_table[0, i] = tot_iter_vi  # Store the number of iterations
        times_table[0, i] = time.time() - start_time  # Store the time taken
        axs[0][i].plot(be_vi, label='VI')
        axs[0][i].set_title(f'VI Init={init_value}')
        axs[0][i].set_xlabel('Iteration')
        axs[0][i].set_ylabel('Bellman Error')
        matches_optimal_table[0, i] = matches_optimal_vi  # Store whether policy matches optimal

        # Plot Q-values as heatmaps for VI
        im = heatmap_axs[0][i].imshow(np.max(Q_vi.reshape((3, 3, n_actions)), axis=-1), cmap='viridis', interpolation='nearest')
        heatmap_axs[0][i].set_title(f'VI Q-Value Function (Init={init_value})')
        for (j, k), val in np.ndenumerate(np.max(Q_vi.reshape((3, 3, n_actions)), axis=-1)):
            heatmap_axs[0][i].annotate(f'{val:.2f}', (k, j), color='white', fontsize=12, ha='center', va='center')

        # Policy Iteration
        start_time = time.time()
        pi_pi, tot_iter_pi, Q_pi, be_pi, matches_optimal_pi = policy_iteration(discount_factor=0.99, Q_init=Q_init)
        tot_iter_table[1, i] = tot_iter_pi  # Store the number of iterations
        times_table[1, i] = time.time() - start_time  # Store the time taken
        axs[1][i].plot(be_pi, label='PI')
        axs[1][i].set_title(f'PI Init={init_value}')
        axs[1][i].set_xlabel('Iteration')
        axs[1][i].set_ylabel('Bellman Error')
        matches_optimal_table[1, i] = matches_optimal_pi  # Store whether policy matches optimal

        # Plot Q-values as heatmaps for PI
        im = heatmap_axs[1][i].imshow(np.max(Q_pi.reshape((3, 3, n_actions)), axis=-1), cmap='viridis', interpolation='nearest')
        heatmap_axs[1][i].set_title(f'PI Q-Value Function (Init={init_value})')
        for (j, k), val in np.ndenumerate(np.max(Q_pi.reshape((3, 3, n_actions)), axis=-1)):
            heatmap_axs[1][i].annotate(f'{val:.2f}', (k, j), color='white', fontsize=12, ha='center', va='center')

        # Generalized Policy Iteration
        start_time = time.time()
        pi_gpi, tot_iter_gpi, Q_gpi, be_gpi, matches_optimal_gpi = generalized_policy_iteration(discount_factor=0.99, Q_init=Q_init)
        tot_iter_table[2, i] = tot_iter_gpi  # Store the number of iterations
        times_table[2, i] = time.time() - start_time  # Store the time taken
        axs[2][i].plot(be_gpi, label='GPI')
        axs[2][i].set_title(f'GPI Init={init_value}')
        axs[2][i].set_xlabel('Iteration')
        axs[2][i].set_ylabel('Bellman Error')
        matches_optimal_table[2, i] = matches_optimal_gpi  # Store whether policy matches optimal

        # Plot Q-values as heatmaps for GPI
        im = heatmap_axs[2][i].imshow(np.max(Q_gpi.reshape((3, 3, n_actions)), axis=-1), cmap='viridis', interpolation='nearest')
        heatmap_axs[2][i].set_title(f'GPI Q-Value Function (Init={init_value})')
        for (j, k), val in np.ndenumerate(np.max(Q_gpi.reshape((3, 3, n_actions)), axis=-1)):
            heatmap_axs[2][i].annotate(f'{val:.2f}', (k, j), color='white', fontsize=12, ha='center', va='center')

    # Compute the mean and standard deviation for each algorithm
    mean_evaluations = np.mean(tot_iter_table, axis=1)
    std_evaluations = np.std(tot_iter_table, axis=1)
    mean_times = np.mean(times_table, axis=1)
    std_times = np.std(times_table, axis=1)

    # Print the results in table format
    print(f"{'Algorithm':<30}{'Mean Evaluations':<20}{'Std Dev Evaluations':<20}{'Mean Time (s)':<20}{'Std Dev Time (s)'}")
    print("-" * 110)
    algorithms = ["Value Iteration", "Policy Iteration", "Generalized Policy Iteration"]
    for i, algo in enumerate(algorithms):
        print(f"{algo:<30}{mean_evaluations[i]:<20.4f}{std_evaluations[i]:<20.4f}{mean_times[i]:<20.4f}{std_times[i]:.4f}")

    # Print the policy match results
    print("\nPolicy Match Results:")
    print(f"{'Algorithm':<30}{'Init Value':<15}{'Matches Optimal Policy'}")
    print("-" * 55)
    for i, algo in enumerate(algorithms):
        for j, init_value in enumerate(init_values):
            print(f"{algo:<30}{init_value:<15}{matches_optimal_table[i, j]}")

    # Display all plots
    plt.tight_layout()
    plt.show()
    heatmap_fig.tight_layout()
    plt.show()

# Run the function
log_and_plot_results_q()

