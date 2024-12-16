import gymnasium
import gym_gridworlds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

working_dir = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT655_RL"
os.chdir(working_dir)


np.set_printoptions(precision=3, suppress=True)

# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x, y):
    return int(0.5 * (x + y) * (x + y + 1) + y)

def rbf_features(x: np.array, c: np.array, s: np.array) -> np.array:
    return np.exp(-(((x[:, None] - c[None]) / s[None])**2).sum(-1) / 2.0)

def aggregation_features(state: np.array, centers: np.array) -> np.array:
    distance = ((state[:, None, :] - centers[None, :, :])**2).sum(-1)
    return (distance == distance.min(-1, keepdims=True)) * 1.0  # make it float

# Coarse features function
def coarse_features(state: np.array, centers: np.array, widths: float, offsets: list = [(0, 0)]) -> np.array:
    """
    Same as tile coding, but we use circles instead of squares, so use the L2
    Euclidean distance to check if a state belongs to a circle.
    Note that coarse coding is more general and allows for ellipses (not just circles)
    but let's consider only circles for the sake of simplicity.
    """

    T = len(offsets)

    # Create shifted centers for each offset
    shifted_centers = np.array([centers + np.array(offset) for offset in offsets])
    
    # Reshape state to match the centers' shape
    state = np.array(state)  # Make sure state is an np.array
    state = state.reshape(-1, 1)  # Reshaping state into (row, col) form if needed
    
    # Calculate the L2 distance between the state and the shifted centers
    distances = np.linalg.norm(state[:, np.newaxis, :] - shifted_centers[:, np.newaxis, :, :], axis=-1)
    
    # Check if state is close to the center within the specified widths
    belongs_to_coarse = distances < widths
    
    # Sum the activations and normalize by the number of offsets
    coarse_activations = belongs_to_coarse.sum(axis=0) / T

    return coarse_activations

def tile_features(state: np.array, centers: np.array, widths: float, offsets: list = [(0, 0)]) -> np.array:
    """
    Given centers and widths, you first have to get an array of 0/1, with 1s
    corresponding to tile the state belongs to.
    If "offsets" is passed, it means we are using multiple tilings, i.e., we
    shift the centers according to the offsets and repeat the computation of
    the 0/1 array. The final output will sum the "activations" of all tilings.
    We recommend to normalize the output in [0, 1] by dividing by the number of
    tilings (offsets).
    Recall that tiles are squares, so you can't use the L2 Euclidean distance to
    check if a state belongs to a tile, but the absolute distance.
    Note that tile coding is more general and allows for rectangles (not just squares)
    but let's consider only squares for the sake of simplicity. 
    """
   
    T = len(offsets)

    shifted_centers = np.array([centers + np.array(offset) for offset in offsets])
    distances = np.abs(state[:, np.newaxis, :] - shifted_centers[:, np.newaxis, :, :]).sum(axis=-1)
    belongs_to_tile = distances < widths
    tile_activations = belongs_to_tile.sum(axis=0) / T

    return tile_activations

def expected_return(env, weights, gamma, episodes=100):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            a = np.dot(phi, weights)
            #a_clip = np.clip(a, env.action_space.low, env.action_space.high)  # this is for the Pendulum
            a = eps_greedy_action(phi, weights, 0)  # this is for the Gridworld
            s_next, r, terminated, truncated, _ = env.step(a)  # replace with a for Gridworld but a_clip for Pendulum
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()


def collect_data(env, weights, sigma, n_episodes):
    data = {"phi": [], "s": [], "a": [], "r": [], "s_next": [], "done": [], "episode_number": []}  # Add episode_number to the dictionary
    for ep in range(n_episodes):
        s, _ = env.reset(seed=ep)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            a = softmax_action(phi, weights, eps)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            data["phi"].append(phi)
            data["s"].append(s)  # Store state
            data["a"].append(a)
            data["r"].append(r)
            data["s_next"].append(s_next)  # Store next state
            data["done"].append(terminated or truncated)
            data["episode_number"].append(ep)  # Store episode number
            s = s_next
            t += 1
    return data

def eps_greedy_action(phi, weights, eps):
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    else:
        Q = np.dot(phi, weights).ravel()
        best = np.argwhere(Q == Q.max())
        i = np.random.choice(range(best.shape[0]))
        return best[i][0]

def softmax_probs(phi, weights, eps):
    q = np.dot(phi, weights)
    # this is a trick to make it more stable
    # see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    q_exp = np.exp((q - np.max(q, -1, keepdims=True)) / max(eps, 1e-12))
    probs = q_exp / q_exp.sum(-1, keepdims=True)
    return probs


def softmax_action(phi, weights, eps):
    probs = softmax_probs(phi, weights, eps)
    return np.random.choice(weights.shape[1], p=probs.ravel())


def gaussian_action(phi: np.array, weights: np.array, sigma: np.array) -> np.array:
    mu = np.dot(phi, weights)
    return np.random.normal(mu, sigma**2)

def dlog_gaussian_probs(phi: np.array, weights: np.array, sigma: float, action: np.array):
    # implement log-derivative of pi with respect to the mean only
    mu = np.dot(phi, weights)
    
    # Log-derivative of Gaussian policy with respect to the mean
    dlog_pi = (action - mu) / (sigma ** 2) * phi
    
    return dlog_pi

def softmax_probs(phi, weights, eps):
    q = np.dot(phi, weights)
    # this is a trick to make it more stable
    # see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    q_exp = np.exp((q - np.max(q, -1, keepdims=True)) / max(eps, 1e-12))
    probs = q_exp / q_exp.sum(-1, keepdims=True)
    return probs

def softmax_action(phi, weights, eps):
    probs = softmax_probs(phi, weights, eps)
    return np.random.choice(weights.shape[1], p=probs.ravel())


def dlog_softmax_probs(phi: np.array, weights: np.array, eps: float, action: int):
    # Compute probabilities using softmax
    probs = softmax_probs(phi, weights, eps)  # Expected shape: (1, n_actions)
    
    # Ensure phi is a 1D array of shape (n_features,)
    phi = phi.ravel()  # Convert to 1D vector if necessary

    # Get the number of features and actions
    n_features = phi.shape[0]
    n_actions = probs.shape[1] if len(probs.shape) > 1 else probs.shape[0]

    # Create a matrix for the gradients
    dlog_pi = np.zeros((n_features, n_actions))

    # Calculate gradients based on softmax probabilities
    for i in range(n_actions):
        if i == action:
            dlog_pi[:, i] = phi * (1 - probs[0, i])  # Gradient for the correct action
        else:
            dlog_pi[:, i] = -phi * probs[0, i]  # Gradient for all other actions

    # Reshape dlog_pi to match the expected shape (1, n_features, n_actions)
    dlog_pi = dlog_pi[np.newaxis, ...]  # Shape: (1, n_features, n_actions)

    return dlog_pi





def estimate_value(env, weights, gamma, episodes=1):
    """
    Estimates the value of a given state using TD(0) method.
    """
    total_value = 0
    
    for e in range(episodes):
        s, _ = env.reset(seed=e)  # Reset the environment and initialize state
        done = False
        discount = 1.0  # Start discount factor at 1 for each episode
        
        while not done:
            phi = get_phi(s).flatten()  # Get state features and ensure they are 1D
            
            a = eps_greedy_action(phi, weights, 0)  # Choose action
            s_next, r, terminated, truncated, _ = env.step(a)  # Step in environment
            done = terminated or truncated
            
            # Accumulate discounted reward
            total_value += discount * r
            
            # Prepare for the next step
            discount *= gamma
            s = s_next
    
    return total_value / episodes  # Return average discounted total value






def mc_return(data, gamma):
    """
    Computes the Monte Carlo (MC) return for each time step.
    
    Parameters:
    - data (dict): Dictionary containing 'r' for rewards and 'done' flags.
    - gamma (float): Discount factor.
    
    Returns:
    - np.array: MC returns for each time step.
    """
    T = len(data["r"])
    returns = np.zeros(T)
    G = 0  # Initialize return G for MC

    # Calculate MC return by iterating backward
    for t in reversed(range(T)):
        r_t = data["r"][t]
        term = data["done"][t]
        G = r_t + (1 - term) * gamma * G
        returns[t] = G
    
    return returns

def mc_return(data, gamma):
    """
    Computes the Monte Carlo (MC) return for each time step.

    Parameters:
    - data (dict): Dictionary containing 'r' for rewards and 'done' flags.
    - gamma (float): Discount factor.

    Returns:
    - np.array: MC returns for each time step.
    """
    T = len(data["r"])
    returns = np.zeros(T)
    G = 0  # Initialize return G for MC

    # Calculate MC return by iterating backward
    for t in reversed(range(T)):
        r_t = data["r"][t]
        term = data["done"][t]  # Check if the episode ended at this step
        if term:  # Reset the return accumulation if the episode ended
            G = 0
        G = r_t + gamma * G
        returns[t] = G
    
    return returns


def collect_data(env, weights, sigma, n_episodes, eps, eps_min=0.1, decay_rate=0.001):
    data = {"phi": [], "s": [], "a": [], "r": [], "s_next": [], "done": [], "episode_number": []}  # Add episode_number to the dictionary
    for ep in range(n_episodes):
        s, _ = env.reset(seed=ep)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            a = softmax_action(phi, weights, eps)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            data["phi"].append(phi)
            data["s"].append(s)  # Store state
            data["a"].append(a)
            data["r"].append(r)
            data["s_next"].append(s_next)  # Store next state
            data["done"].append(done)
            data["episode_number"].append(ep)  # Store episode number
            s = s_next
            t += 1
            eps = max(eps_min, eps - decay_rate)
    return data




def td_error_return(data, weights, n_steps=1, alpha_weights=0.1, gamma=0.99):
    """
    Compute TD errors and update weights using n-step TD learning for the selected action.
    
    Parameters:
    - data (dict): Collected data with 'phi', 'r', 'a', 'done'.
    - weights (np.ndarray): Weight matrix for the state-action value function.
    - n_steps (int): Number of steps for n-step TD.
    - alpha_weights (float): Learning rate for weight updates.
    - gamma (float): Discount factor.
    
    Returns:
    - computed_rewards (list): List of n-step returns.
    - weights (np.ndarray): Updated weights.
    - td_errors (list): List of TD errors.
    """
    phi_values = np.array(data["phi"])  # Convert to NumPy array (T, features)
    rewards = data["r"]
    actions = data["a"]
    done_flags = data["done"]
    T = len(rewards)

    td_errors = []
    computed_rewards = []

    for t in range(T):
        tau = t - n_steps + 1

        if tau >= 0:
            # Get the feature vector and action for time step tau
            current_phi = phi_values[tau].flatten()
            action_tau = actions[tau]
            V_s_t = np.max(np.dot(current_phi, weights))  # Value of the current state-action pair

            # Compute the n-step return
            G = 0
            for i in range(tau, min(tau + n_steps, T)):
                G += (gamma ** (i - tau)) * rewards[i]
                if done_flags[i]:  # Stop summing if terminal state is reached
                    break

            # Add the value of the next state if within bounds and not terminal
            if tau + n_steps < T and not done_flags[tau + n_steps - 1]:
                next_phi = phi_values[tau + n_steps].flatten()
                V_s_tp1 = np.max(np.dot(next_phi, weights))  # Max Q-value for the next state
                G += (gamma ** n_steps) * V_s_tp1
            else:
                G = G

            # Compute the TD error
            td_error_t = G - V_s_t
            td_errors.append(td_error_t)

            # Update weights for the action at time step tau
            weights[:, action_tau] += alpha_weights * td_error_t * current_phi

            # Store the n-step return
            computed_rewards.append(G)

    return computed_rewards, weights, td_errors



def reinforce(method="none", n_steps=1):
    """
    Performs REINFORCE (policy gradient) with Monte Carlo, Temporal Difference, or n-step updates.

    Parameters:
    - method (str): Method to use for estimating returns: "mc", "td_zero", "n_step", or "none".
    - gamma (float): Discount factor.
    - max_steps (int): Maximum number of steps to run.
    - episodes_per_update (int): Number of episodes per gradient update.
    - alpha (float): Learning rate for updating weights.
    - env: The environment object for training.
    - env_eval: The environment object for evaluation.
    
    Returns:
    - exp_return_history (np.array): History of expected returns.
    - performance_variance_history (np.array): History of performance variance.
    - weight_variance_history (np.array): History of weight variance.
    - gradient_variance_history (np.array): History of gradient variance.
    """

    output_dir="results"
    os.makedirs(output_dir, exist_ok=True)
    weights = np.zeros((phi_dummy.shape[1], action_dim))
    theta = np.zeros((phi_dummy.shape[1], action_dim))
    sigma = 2.0
    
    exp_return = expected_return(env_eval, theta, gamma, episodes_eval)
    exp_value = estimate_value(env_eval, theta, gamma, episodes_eval)
    
    pbar = tqdm(total=max_steps)
    gradients_list = []

    # Initialize the dictionary to collect all metrics
    # Initialize history arrays for the four metrics
    exp_return_history = np.zeros(max_steps)
    weight_variance_history = np.zeros(max_steps)
    theta_variance_history = np.zeros(max_steps)
    performance_variance_history = np.zeros(max_steps)
    gradient_variance_history = np.zeros(max_steps)
    td_error_history = np.zeros(max_steps)
    reward_history = np.zeros(max_steps)
    time_steps_history = np.zeros(max_steps)
    average_return_history = np.zeros(max_steps)
    episode_history = np.zeros(max_steps)
    
    metrics = {
        "step": time_steps_history,
        "reward": reward_history,
        "average_return": average_return_history,
        "expectation": exp_return_history,
        "gradient_entropy": gradient_variance_history,
        "weights_stability": weight_variance_history,
        "theta_stability": theta_variance_history,
        "performance": performance_variance_history,
        "td_error": td_error_history,
        "episode": episode_history,
    }

    tot_steps = 0
    tot_episodes = 0 
    while tot_steps < max_steps:
        data = collect_data(env, theta, sigma, episodes_per_update, eps)
        T = len(data["r"])
        td_errors = np.zeros(T)
        term = data["done"]

        # Update buffers
        if method == "mc":
            returns = mc_return(data, gamma)
            baseline_value = returns.mean()  # Baseline as average MC return
            
        elif method == "td_zero":
            returns, weights, td_errors = td_error_return(data, weights, n_steps=1, alpha_weights=alpha_weights, gamma=gamma)
        elif method.startswith("n_step"):
            n_steps_value = int(method.split()[1])  # Extract n_step value from method string
            returns, weights, td_errors = td_error_return(data, weights, n_steps=n_steps_value, alpha_weights=alpha_weights, gamma=gamma) 
        else:
            raise ValueError("Invalid method for reinforce")

        # Performance variance calculation
        average_return = np.mean(returns)
        performance_variance = np.var(returns)
        theta_variance = np.var(theta)
        weight_variance = np.var(weights)

        num_episodes = len(np.unique(data["episode_number"])) 
        # episode_offset = tot_episodes
        # adjusted_episode_numbers = [
        #     ep + episode_offset for ep in data["episode_number"]
        # ]
        # Ensure T doesn't exceed max_steps
        if tot_steps + T > max_steps:
            T = max_steps - tot_steps

        # Store performance variance history
        performance_variance_history[tot_steps: tot_steps + T] = performance_variance
        reward_history[tot_steps: tot_steps + T] = data["r"][:T]
        average_return_history[tot_steps: tot_steps + T] = average_return

        episode_history[tot_steps: tot_steps + T] = np.array([tot_episodes] * T)
        tot_episodes += num_episodes  

        #print(data["episode_number"][:T])
        theta_variance_history[tot_steps: tot_steps + T] = theta_variance
        time_steps_history[tot_steps: tot_steps + T] = np.arange(tot_steps, tot_steps + T)
        weight_variance_history[tot_steps: tot_steps + T] = weight_variance

        # Calculate gradients
        action_matrix = -1 * np.ones((n_row, n_col), dtype=int)
        for t in range(T):
            phi_t = data["phi"][t]
            a_t = data["a"][t]
            

            #PRINTING THE ACTION PER STATE
            state_idx = data["s"][t]  # Assuming `s` contains state indices
            row, col = int(state_idx[0]), int(state_idx[1])
            # Update action matrix if within bounds
            if 0 <= row < n_row and 0 <= col < n_col:
                action_matrix[row, col] = a_t

            # #PRINTING THE ACTION PER STATE
            # state_idx = data["s"][t]  # Assuming `s` contains state indices
            # state_coordinates = state_to_coordinates(state_idx)  # Convert to (row, col)
            #  # Update the action matrix
            # row, col = state_coordinates  # Unpack coordinates
            # if 0 <= row < action_matrix.shape[0] and 0 <= col < action_matrix.shape[1]:
            #     action_matrix[row, col] = a_t  # Update with action
            
            if method == "mc":
                # Gradient update with Monte Carlo return
                G_t = returns[t]
                delta = G_t - baseline_value
                grad_log_pi = dlog_softmax_probs(phi_t, theta, eps, a_t)
                theta += alpha_theta * (gamma ** t) * delta * grad_log_pi.reshape(theta.shape)
                gradients_list.append(grad_log_pi)

            elif method == "td_zero":
                # Gradient update with TD(0) error
                delta = td_errors[t]  # TD(0) uses all timesteps
                grad_log_pi = dlog_softmax_probs(phi_t, theta, eps, a_t).ravel()
                theta += alpha_theta * (gamma ** t) * delta * grad_log_pi.reshape(theta.shape)
                gradients_list.append(grad_log_pi)
                td_error_history[tot_steps + t] = td_errors[t]

            elif method.startswith("n_step"):
                # Gradient update with n-step TD
                n_steps_value = int(method.split()[1])  # Extract n-step value
                tau = t - n_steps_value + 1  # Adjust index for n-step condition

                if tau >= 0:  # Only update for valid tau
                    phi_tau = data["phi"][tau]
                    a_tau = data["a"][tau]
                    delta = td_errors[tau]  # Use TD error at adjusted index
                    grad_log_pi = dlog_softmax_probs(phi_tau, theta, eps, a_tau).ravel()
                    theta += alpha_theta * (gamma ** tau) * delta * grad_log_pi.reshape(theta.shape)
                    gradients_list.append(grad_log_pi)
                    td_error_history[tot_steps + tau] = td_errors[tau]

            else:
                raise ValueError("Invalid method specified for gradient updates")

        # Calculate gradient variance after accumulating enough gradients
        if len(gradients_list) > 1:
            gradient_variance = np.var(gradients_list, axis=0)
            gradient_variance_history[tot_steps: tot_steps + T] = gradient_variance.mean()

        # Expectations
        if method == "mc":
            exp_return_history[tot_steps: tot_steps + T] = exp_return
            exp_return = expected_return(env_eval, theta, gamma, episodes_eval)
        else:
            if tot_steps + T <= max_steps:
                exp_return_history[tot_steps: tot_steps + T] = exp_return
                exp_return = estimate_value(env_eval, theta, gamma, episodes_eval)

        tot_steps += T
        sigma = max(sigma - T / max_steps, 0.1)
        

        pbar.set_description(

            f"Method: {method} G: {exp_return:.3f}"
        )
        pbar.update(T)

    # Store the metrics
    metrics = {
        "step": time_steps_history[:tot_steps],
        "reward": reward_history[:tot_steps],
        "average_return": average_return_history[:tot_steps],
        "expectation": exp_return_history[:tot_steps],
        "gradient_entropy": gradient_variance_history[:tot_steps],
        "weights_stability": weight_variance_history[:tot_steps],
        "theta_stability": theta_variance_history[:tot_steps],
        "performance": performance_variance_history[:tot_steps],
        "td_error": td_error_history[:tot_steps],
        "episode": episode_history[:tot_steps],
    }
    #print(episode_history[:tot_steps])
    #print(get_policy(theta))
    print("Action Matrix:")
    print(action_matrix)

    return metrics



# env_id = "Pendulum-v1"
# env = gymnasium.make(env_id)
# env_eval = gymnasium.make(env_id)
# episodes_eval = 100
# # you'll solve the Pendulum when the empirical expected return is higher than -150
# # but it can get even higher, eg -120
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]

# UNCOMMENT TO SOLVE THE GRIDWORLD
#env_id = "Gym-Gridworlds/Penalty-3x3-v0"
env_id = "Gym-Gridworlds/CliffWalk-4x12-v0"
#env_id = "Gym-Gridworlds/DangerMaze-6x6-v0"
env = gymnasium.make(env_id, coordinate_observation=True, distance_reward=True, max_episode_steps=10000)
#env = gymnasium.make(env_id, coordinate_observation=True, distance_reward=True, max_episode_steps=10000, reward_noise_std=0.05)
#env = gymnasium.make(env_id, coordinate_observation=True, distance_reward=True, max_episode_steps=10000, random_action_prob=0.1)
#env = gymnasium.make(env_id, coordinate_observation=True, distance_reward=True, max_episode_steps=10000, observation_noise=0.2)
#env = gymnasium.make(env_id, coordinate_observation=True, distance_reward=True, max_episode_steps=10000, random_action_prob=0.1, reward_noise_std=0.05, observation_noise=0.2)
env_eval = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=10)  # 10 steps only for faster eval
episodes_eval = 1  # max expected return will be 0.941
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
action_dim = env.action_space.n


#automatically set centers and sigmas
n_centers = [7] * state_dim
state_low = env.observation_space.low
state_high = env.observation_space.high
centers = np.array(
    np.meshgrid(*[
        np.linspace(
            state_low[i] - (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
            state_high[i] + (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
            n_centers[i],
        )
        for i in range(state_dim)
    ])
).reshape(state_dim, -1).T
sigmas = (state_high - state_low) / np.asarray(n_centers) * 0.75 + 1e-8  # change sigmas for more/less generalization
get_phi = lambda state : rbf_features(state.reshape(-1, state_dim), centers, sigmas)  # reshape because feature functions expect shape (N, S)
#get_phi = lambda state : aggregation_features(state.reshape(-1, state_dim), centers)
phi_dummy = get_phi(env.reset()[0])  # to get the number of features


# ##env_id = "FrozenLake-v1"
# env_id = "Gym-Gridworlds/Penalty-3x3-v0"
# reward_noise_sd = 0.0
# rand_act_prob = 0.0
# obs_noise = 0.0
# env = gymnasium.make(env_id, coordinate_observation=True,distance_reward=True, max_episode_steps=1000, reward_noise_std = reward_noise_sd, random_action_prob = rand_act_prob, observation_noise = obs_noise)
# # env_copy = gymnasium.make(env_id, coordinate_observation=True,distance_reward=True, max_episode_steps=1000, reward_noise_std = 0.0, random_action_prob = 0.0, observation_noise = 0.0)
# #env = gymnasium.make(env_id, coordinate_observation=True) # changed max_episode_steps
# env_eval = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=10)  # 10 steps only for faster eval
# #env_eval = gymnasium.make(env_id, coordinate_observation=True) 
# episodes_eval = 1 # max expected return will be 0.941
# state_dim = env.observation_space.shape[0]
# n_actions = env.action_space.n
# action_dim = env.action_space.n


# # automatically set centers and sigmas
# n_centers = [9] * state_dim
# #n_centers = [20, 20]
# print(n_centers)
# state_low = env.observation_space.low
# state_high = env.observation_space.high
# centers = np.array(
#     np.meshgrid(*[
#         np.linspace(
#             state_low[i] - (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
#             state_high[i] + (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
#             n_centers[i],
#         )
#         for i in range(state_dim)
#     ])
# ).reshape(state_dim, -1).T
# sigmas = (state_high - state_low) / np.asarray(n_centers) * 0.75 + 1e-8  # change sigmas for more/less generalization
# #get_phi = lambda state : rbf_features(state.reshape(-1, state_dim), centers, sigmas)  # reshape because feature functions expect shape (N, S)
# get_phi = lambda state : aggregation_features(state.reshape(-1, state_dim), centers)
# phi_dummy = get_phi(env.reset()[0])  # to get the number of features




n_row = 3
n_col = 3
def get_policy(weights):
    pi = []
    for row in range(n_row):
        new_row = []
        for col in range(n_col):
            phi = get_phi(np.array([row, col]))
            new_row.append(eps_greedy_action(phi, weights, 0))
        pi.append(new_row)
    return np.array(pi)

def state_to_coordinates(state_idx, grid_shape=(n_row, n_col)):
    """
    Converts a 1D state index to a 2D coordinate (row, col) for a grid.
    """
    #print(f"Grid: {grid_shape}")
    rows, cols = grid_shape
    row = state_idx // cols  # Integer division for row
    col = state_idx % cols  # Modulus for column
    return row, col

def state_to_coordinates(state, centers=centers, grid_shape=(n_row, n_col)):
    """
    Maps a continuous state to a grid coordinate (row, col).
    """
    # Find the closest center to the state
    closest_center_idx = np.argmin(np.linalg.norm(centers - state, axis=1))
    row, col = divmod(closest_center_idx, grid_shape[1])  # Convert to grid coordinates
    return row, col

# Experiment setup with baselines and seeds
gamma = 0.99
alpha_weights = 0.1
alpha_theta = 0.01
eps = 1.0
episodes_per_update = 10
max_steps = 10000
n_seeds = 3



# Define the methods
#methods = ["mc", "td_zero", "td_lambda", "n_step"]
#methods = ["mc", "td_zero", "n_step"]
methods = ["mc", "td_zero", "n_step 2", "n_step 3", "n_step 4"]


# Initialize dictionary to store metrics for all methods, seeds, and steps
all_metrics = {metric: np.zeros((len(methods), n_seeds, max_steps)) for metric in
               ["step", "reward", "average_return", "expectation", "gradient_entropy", "weights_stability", "theta_stability", "performance", "td_error", "episode"]}

# Run reinforce once and collect all metrics
for i, method in enumerate(methods):
    n_steps_value = int(method.split()[1]) if method.startswith("n_step") else 1
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        # Run reinforce and get metrics dictionary
        metrics = reinforce(method, n_steps=n_steps_value)

        # Store metrics in the all_metrics dictionary (no normalization)
        for key, value in metrics.items():
            all_metrics[key][i, seed] = value


# Save averaged metrics to file
output_dir = "results"
#NoiseObservePenalty, NoiseTransPenalty, NoiseRewardPenalty, Penalty
prefix_name = "NoiseTransPenalty"
os.makedirs(output_dir, exist_ok=True)

for method_idx, method in enumerate(methods):
    file_path = os.path.join(output_dir, f"{prefix_name}_{method}_metrics.npz")
    np.savez(file_path, **{metric: all_metrics[metric][method_idx] for metric in all_metrics})


from plots_RL_metrics import smooth
from plots_RL_metrics import error_shade_plot
from plots_RL_metrics import  plot_metrics_against_episode_and_timestep

plot_metrics_against_episode_and_timestep(results_dir=output_dir, prefix=prefix_name, normalize=True)
