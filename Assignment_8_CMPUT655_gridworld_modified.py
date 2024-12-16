import gymnasium
import gym_gridworlds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x, y):
    return int(0.5 * (x + y) * (x + y + 1) + y)

def rbf_features(x: np.array, c: np.array, s: np.array) -> np.array:
    return np.exp(-(((x[:, None] - c[None]) / s[None])**2).sum(-1) / 2.0)

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
    data = dict()
    data["phi"] = []
    data["a"] = []
    data["r"] = []
    data["done"] = []
    for ep in range(n_episodes):
        episode_seed = cantor_pairing(ep, seed)
        s, _ = env.reset(seed=episode_seed)
        done = False
        while not done:
            phi = get_phi(s)
            #a = gaussian_action(phi, weights, sigma)
            a = softmax_action(phi, weights, eps)
            #a_clip = np.clip(a, env.action_space.low, env.action_space.high)  # only for Gaussian policy
            s_next, r, terminated, truncated, _ = env.step(a) #change to a for gridworld but a_clip for Pendulum
            done = terminated or truncated
            data["phi"].append(phi)
            data["a"].append(a)
            data["r"].append(r)
            data["done"].append(terminated or truncated)
            s = s_next
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


# Define normalization functions
# Define normalization functions with debug statements
def z_score_normalize(data):
    print(f"Original shape for z-score normalization: {data.shape}")  # Debug statement
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std if std != 0 else data
    print(f"Normalized shape for z-score normalization: {normalized_data.shape}")  # Debug statement
    return normalized_data

def min_max_scale(data):
    print(f"Original shape for min-max scaling: {data.shape}")  # Debug statement
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val) if max_val != min_val else data
    print(f"Scaled shape for min-max scaling: {scaled_data.shape}")  # Debug statement
    return scaled_data

# Reinforce function with variance tracking
import numpy as np
from tqdm import tqdm

# Reinforce function with variance tracking
# Reinforce function with variance tracking
def reinforce(baseline="none"):
    weights = np.zeros((phi_dummy.shape[1], action_dim))
    sigma = 2.0
    eps = 1.0
    tot_steps = 0
    exp_return_history = np.zeros(max_steps)
    weight_variance_history = np.zeros(max_steps)
    performance_variance_history = np.zeros(max_steps)
    gradient_variance_history = np.zeros(max_steps)
    pbar = tqdm(total=max_steps)
    alpha = 0.1
    baseline_value = 0
    gradients_list = []

    while tot_steps < max_steps:
        data = collect_data(env, weights, sigma, episodes_per_update)
        T = len(data["r"])
        
        # Compute returns and track performance variance
        G = 0
        returns = np.zeros(T)
        for t in reversed(range(T)):
            r_t = data["r"][t]
            term = data["done"][t]
            G = r_t + (1 - term) * gamma * G
            returns[t] = G
            
        # Debug: Check the shape of returns
        print(f"Returns shape: {returns.shape}")
        
        # Performance variance calculation
        performance_variance = np.var(returns)
        print(f"Performance variance: {performance_variance}, Shape: {performance_variance.shape}")

        # Store performance variance history
        if tot_steps + T <= max_steps:
            performance_variance_history[tot_steps: tot_steps + T] = performance_variance * np.ones(T)
        else:
            print(f"Warning: Attempting to write out of bounds in performance_variance_history at tot_steps={tot_steps}, T={T}.")

        # Baseline adjustment
        if baseline == "mean_return":
            baseline_value = returns.mean()
        elif baseline == "optimal":
            gradient_squares = np.zeros(T)
            weighted_returns = np.zeros(T)
            for t in range(T):
                phi_t = data["phi"][t]
                a_t = data["a"][t]
                dlog_pi = dlog_softmax_probs(phi_t, weights, eps, a_t)
                grad_squared = np.sum(dlog_pi ** 2)
                gradient_squares[t] = grad_squared
                weighted_returns[t] = grad_squared * returns[t]
            baseline_value = (weighted_returns.sum() / gradient_squares.sum()
                              if gradient_squares.sum() != 0 else 0)

        # Calculate and store gradients for gradient variance
        gradient = np.zeros((T, len(weights.ravel())))
        for t in range(T):
            phi_t = data["phi"][t]
            a_t = data["a"][t]
            G_t = returns[t]
            delta = G_t - baseline_value
            dlog_pi = dlog_softmax_probs(phi_t, weights, eps, a_t)
            gradient[t] = delta * dlog_pi.ravel()
        
        gradients_list.append(gradient.mean(0))
        
        # Debug: Check the shape of the gradient list
        print(f"Gradient shape: {gradient.shape}, Gradients List length: {len(gradients_list)}")

        if len(gradients_list) > 1:
            gradient_variance = np.var(gradients_list, axis=0)
            gradient_variance_history[tot_steps: tot_steps + T] = gradient_variance.mean()
            # Debug: Check the shape of the gradient variance
            print(f"Gradient variance shape: {gradient_variance.shape}")

        # Update weights and track weight variance
        weights += alpha * gradient.mean(0).reshape(weights.shape)
        weight_variance = np.var(weights)
        weight_variance_history[tot_steps: tot_steps + T] = weight_variance
        # Debug: Check the weight variance
        print(f"Weight variance: {weight_variance}")

        # Evaluate expected return
        exp_return_history[tot_steps: tot_steps + T] = expected_return(env_eval, weights, gamma, episodes_eval)
        tot_steps += T
        sigma = max(sigma - T / max_steps, 0.1)
        if tot_steps - 1 < len(exp_return_history):
            pbar.set_description(f"G: {exp_return_history[tot_steps - 1]:.3f}")
        else:
            print("Warning: tot_steps is out of bounds.")

        pbar.update(T)

    pbar.close()
    return exp_return_history, performance_variance_history, weight_variance_history, gradient_variance_history


def reinforce(baseline="none"):
    weights = np.zeros((phi_dummy.shape[1], action_dim))
    sigma = 2.0
    tot_steps = 0
    exp_return_history = np.zeros(max_steps)
    weight_variance_history = np.zeros(max_steps)
    performance_variance_history = np.zeros(max_steps)
    gradient_variance_history = np.zeros(max_steps)
    pbar = tqdm(total=max_steps)
    alpha = 0.1
    baseline_value = 0
    gradients_list = []

    while tot_steps < max_steps:
        data = collect_data(env, weights, sigma, episodes_per_update)
        T = len(data["r"])
        
        # Compute returns and track performance variance
        G = 0
        returns = np.zeros(T)
        for t in reversed(range(T)):
            r_t = data["r"][t]
            term = data["done"][t]
            G = r_t + (1 - term) * gamma * G
            returns[t] = G
        
        # Performance variance calculation
        performance_variance = np.var(returns)

        # Store performance variance history
        if tot_steps + T <= max_steps:
            performance_variance_history[tot_steps: tot_steps + T] = performance_variance * np.ones(T)

        # Baseline adjustment
        if baseline == "mean_return":
            baseline_value = returns.mean()
        elif baseline == "optimal":
            gradient_squares = np.zeros(T)
            weighted_returns = np.zeros(T)
            for t in range(T):
                phi_t = data["phi"][t]
                a_t = data["a"][t]
                dlog_pi = dlog_softmax_probs(phi_t, weights, eps, a_t)
                grad_squared = np.sum(dlog_pi ** 2)
                gradient_squares[t] = grad_squared
                weighted_returns[t] = grad_squared * returns[t]
            baseline_value = (weighted_returns.sum() / gradient_squares.sum()
                              if gradient_squares.sum() != 0 else 0)

        # Calculate gradients
        gradient = np.zeros((T, len(weights.ravel())))
        for t in range(T):
            phi_t = data["phi"][t]
            a_t = data["a"][t]
            G_t = returns[t]
            delta = G_t - baseline_value
            dlog_pi = dlog_softmax_probs(phi_t, weights, eps, a_t)
            gradient[t] = delta * dlog_pi.ravel()
        
        gradients_list.append(gradient.mean(0))

        # Calculate gradient variance after accumulating enough gradients
        if len(gradients_list) > 1:
            gradient_variance = np.var(gradients_list, axis=0)
            gradient_variance_history[tot_steps: tot_steps + T] = gradient_variance.mean()

        # Update weights
        weights += alpha * gradient.mean(0).reshape(weights.shape)
        weight_variance = np.var(weights)
        weight_variance_history[tot_steps: tot_steps + T] = weight_variance

        # Evaluate expected return
        exp_return_history[tot_steps: tot_steps + T] = expected_return(env_eval, weights, gamma, episodes_eval)
        tot_steps += T
        sigma = max(sigma - T / max_steps, 0.1)
        pbar.update(T)

    pbar.close()
    return exp_return_history, performance_variance_history, weight_variance_history, gradient_variance_history





# env_id = "Pendulum-v1"
# env = gymnasium.make(env_id)
# env_eval = gymnasium.make(env_id)
# episodes_eval = 100
# # you'll solve the Pendulum when the empirical expected return is higher than -150
# # but it can get even higher, eg -120
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]

# UNCOMMENT TO SOLVE THE GRIDWORLD
env_id = "Gym-Gridworlds/Penalty-3x3-v0"
env = gymnasium.make(env_id, coordinate_observation=True, distance_reward=True, max_episode_steps=10000)
env_eval = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=10)  # 10 steps only for faster eval
episodes_eval = 1  # max expected return will be 0.941
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
action_dim = env.action_space.n


# automatically set centers and sigmas
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
phi_dummy = get_phi(env.reset()[0])  # to get the number of features

# Experiment setup with baselines and seeds
gamma = 0.99
alpha = 0.1
eps = 1.0
episodes_per_update = 10
max_steps = 1000
baselines = ["none", "mean_return", "optimal"]
n_seeds = 2
results_exp_ret = np.zeros((len(baselines), n_seeds, max_steps))



# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    print("Data shape:", data.shape)  # Debugging line
    print(f"DATA: {data[:50]}")
    print(f"Data type before processing: {type(data)}")
    print(f"Data dtype before processing: {data.dtype if hasattr(data, 'dtype') else 'No dtype'}")
    print(f"Data shape before processing: {data.shape}")
    
    if data.ndim == 1:
        data = data.reshape(1, -1)  # Reshape 1D array to 2D
    
    if data.ndim != 2:
        raise ValueError("Expected a 2D array for 'data', but got a {}D array.".format(data.ndim))
    
    y = np.nanmean(data, 0)  # Calculate the mean along the specified axis
    if np.isscalar(y):  # If y is a scalar, raise an error
        raise ValueError("Computed mean is a scalar, expected an array.")

    x = np.arange(len(y))  # x should match the length of y
    x = [stepsize * step for step in range(len(y))]  # Scale x appropriately
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())



import matplotlib.pyplot as plt
import numpy as np

# Assuming `baselines`, `n_seeds`, `results_exp_ret`, `reinforce`, `z_score_normalize`, `min_max_scale`, and `error_shade_plot` are defined

# Initialize lists to store results
import matplotlib.pyplot as plt
import numpy as np

# Assuming `baselines`, `n_seeds`, `results_exp_ret`, `reinforce`, `z_score_normalize`, `min_max_scale`, and `error_shade_plot` are defined

# Initialize lists to store results
results_exp_ret = np.zeros((len(baselines), n_seeds, max_steps))  # Adjust the second dimension as needed for expected return history
perf_var_matrix = np.zeros((len(baselines), n_seeds, max_steps))  # Adjust shape to hold normalized values
weight_var_matrix = np.zeros((len(baselines), n_seeds, max_steps))
grad_var_matrix = np.zeros((len(baselines), n_seeds, max_steps))

# Run reinforce once and collect all metrics
for i, baseline in enumerate(baselines):
    for seed in range(n_seeds):
        np.random.seed(seed)

        # Run reinforce and collect metrics
        exp_return_history, perf_var, weight_var, grad_var = reinforce(baseline)
        results_exp_ret[i, seed] = exp_return_history
        
        # Store the normalized variance metrics
        perf_var_normalized = z_score_normalize(perf_var).flatten()  # Ensure 1D
        weight_var_normalized = min_max_scale(weight_var).flatten()  # Ensure 1D
        grad_var_normalized = z_score_normalize(grad_var).flatten()  # Ensure 1D

        # Store normalized arrays for later averaging
        perf_var_matrix[i, seed] = perf_var_normalized
        weight_var_matrix[i, seed] = weight_var_normalized
        grad_var_matrix[i, seed] = grad_var_normalized

# Now, average over the seeds for each baseline
perf_var_avg = np.mean(perf_var_matrix, axis=1)  # Average over seeds
weight_var_avg = np.mean(weight_var_matrix, axis=1)
grad_var_avg = np.mean(grad_var_matrix, axis=1)

# Debugging: Check the shapes of averaged metrics
print(f"perf_var_avg shape: {perf_var_avg.shape}")
print(f"weight_var_avg shape: {weight_var_avg.shape}")
print(f"grad_var_avg shape: {grad_var_avg.shape}")

# Now proceed with plotting the averages
# Plot for Expected Return
plt.figure(figsize=(10, 5))
plt.title("Expected Return for Baselines")
for i, baseline in enumerate(baselines):
    error_shade_plot(plt.gca(), results_exp_ret[i], stepsize=1, smoothing_window=20, label=baseline)

plt.xlabel("Steps")
plt.ylabel("Expected Return")
plt.legend()
plt.tight_layout()
plt.show()

# Plot for Performance Variance (Z-score Normalized)
plt.figure(figsize=(10, 5))
plt.title("Performance Variance (Z-score Normalized)")
for i in range(len(perf_var_avg)):
    error_shade_plot(plt.gca(), perf_var_avg[i], stepsize=1, smoothing_window=20, label=f"{baselines[i]} (Z-score)")  # Adjusted index for baselines

plt.xlabel("Steps")
plt.ylabel("Performance Variance")
plt.legend()
plt.tight_layout()
plt.show()

# Plot for Weight Variance (Min-Max Scaled)
plt.figure(figsize=(10, 5))
plt.title("Weight Variance (Min-Max Scaled)")
for i in range(len(weight_var_avg)):
    error_shade_plot(plt.gca(), weight_var_avg[i], stepsize=1, smoothing_window=20, label=f"{baselines[i]} (Min-Max)")  # Adjusted index for baselines

plt.xlabel("Steps")
plt.ylabel("Weight Variance")
plt.legend()
plt.tight_layout()
plt.show()

# Plot for Gradient Variance (Z-score Normalized)
plt.figure(figsize=(10, 5))
plt.title("Gradient Variance (Z-score Normalized)")
for i in range(len(grad_var_avg)):
    error_shade_plot(plt.gca(), grad_var_avg[i], stepsize=1, smoothing_window=20, label=f"{baselines[i]} (Z-score)")  # Adjusted index for baselines

plt.xlabel("Steps")
plt.ylabel("Gradient Variance")
plt.legend()
plt.tight_layout()
plt.show()


