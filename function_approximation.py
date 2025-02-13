#import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import numpy as np

def poly_features(state: np.array, degree: int) -> np.array:
    """
    Compute polynomial features. For example, if state = (s1, s2) and degree = 2,
    the output must be 1 + s1 + s2 + s1*s2 + s1**2 + s2**2
    """
    state = state
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    features = poly.fit_transform(state)
    return features

def rbf_features(state: np.array, centers: np.array, sigmas: float) -> np.array:
    """
    Compute radial basis functions features: exp(- ||s - c||**2 / (2 * sigma**2)).
    s is the state, c are the centers, and sigma is the width of the Gaussian.
    """
    # Compute squared Euclidean distances ||s - c||^2 for each pair of state and center
    distances = np.linalg.norm(state[:, np.newaxis, :] - centers, axis=2) ** 2
    # Compute RBF features for each center (Gaussian radial basis function)
    rbf_features = np.exp(-distances / (2 * sigmas ** 2))
    return rbf_features

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
    distances = np.linalg.norm(state[:, np.newaxis, :] - shifted_centers[:, np.newaxis, :, :], axis=-1)
    belongs_to_coarse = distances < widths
    coarse_activations = belongs_to_coarse.sum(axis=0) / T

    return coarse_activations

def aggregation_features(state: np.array, centers: np.array) -> np.array:
    """
    Aggregate states to the closest center. The output will be an array of 0s and
    one 1 corresponding to the closest tile the state belongs to.
    Note that we can turn this into a discrete (finite) representation of the state,
    because we will have as many feature representations as centers.
    """
    distances = np.linalg.norm(state[:, np.newaxis, :] - centers, axis=2)
    closest_tile_indices = np.argmin(distances, axis=1)
    agg_coding = np.zeros((state.shape[0], centers.shape[0]), dtype=int)
    agg_coding[np.arange(state.shape[0]), closest_tile_indices] = 1
    return agg_coding
    

# Assume tile_features, coarse_features, and aggregation_features functions remain unchanged

data = np.load("/media/adebayo/Windows/UofACanada/study/codes/CMPUT655_RL/a6_gridworld.npz")
s = data["s"]
a = data["a"]
r = data["r"]
s_next = data["s_next"]
Q = data["Q"]
V = data["Q"].max(-1)
term = data["term"]
n = s.shape[0]

state_size = 2
n_samples = 10
n_centers = 100
state = np.random.rand(n_samples, state_size)

state_1_centers = np.linspace(-10, 10, n_centers)
state_2_centers = np.linspace(-10, 10, n_centers)
centers = np.array(
    np.meshgrid(state_1_centers, state_2_centers)
).reshape(state_size, -1).T

widths = 0.5
sigmas = 0.5

fig, axs = plt.subplots(1, 1)
axs.tricontourf(s[:, 0], s[:, 1], V, levels=100)
plt.show()

max_iter = 10000
gamma = 0.99
alpha = 1.0
thresh = 1e-8
offsets = [(-0.1, 0.0), (0.0, 0.1), (0.1, 0.0), (0.0, -0.1)]

# Pick one feature representation method
#name, get_phi = "RBFs", lambda state: rbf_features(state, centers, sigmas)
name, get_phi = "coarse_features", lambda state : tile_features(state, centers, widths, offsets)

phi = get_phi(s)
phi_next = get_phi(s_next)

num_features = phi.shape[1]  # Number of features
num_actions = 5  # Number of possible actions in the environment

# Initialize weights as a 2D array with shape (num_features, num_actions)
weights = np.zeros((num_features, num_actions))

pbar = tqdm(total=max_iter)
epsilon = 0.1  # Epsilon for epsilon-greedy policy

def epsilon_greedy_policy(q_values, epsilon=0.1):
    """
    Epsilon-greedy policy to select action based on Q-values.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))  # Random action
    else:
        return np.argmax(q_values)  # Best action based on Q-values

for iter in range(max_iter):
    # Compute Q-values for current state-action pairs
    q_current = np.zeros((n, num_actions))
    q_predict = np.zeros((n, num_actions))  # For next state-action pairs
    
    # Iterate over each action
    for action in range(num_actions):
        q_current[:, action] = phi @ weights[:, action]  # Q(s, a) for current state
        q_predict[:, action] = phi_next @ weights[:, action]  # Q(s', a') for next state

    # Use epsilon-greedy policy to select the next action
    next_actions = np.array([epsilon_greedy_policy(q_predict[i], epsilon) for i in range(n)])

    # Compute the TD target using the next state-action pair
    td_target = r + (1 - term) * gamma * q_predict[np.arange(n), next_actions]

    # TD error for Q-learning
    td_error = td_target - q_current[np.arange(n), a]  # Using current action a

    # Calculate Mean Squared Error (MSE)
    mse = np.mean(td_error**2)

    # Update the weights using the semi-gradient update rule
    for action in range(num_actions):
        weights[:, action] += alpha * (phi.T @ (td_error * (a == action))) / len(s)

    # Reduce alpha (learning rate) over time
    epsilon = max(epsilon - epsilon/max_iter, 0.1)
    alpha = max(alpha - 1 / max_iter, 0.001)

    pbar.set_description(f"TDE: {td_error}, MSE: {mse}")
    pbar.update()

    if mse < thresh:
        break

# Calculate the approximated Q-values using the learned weights
approx_q = np.zeros((n, num_actions))
for action in range(num_actions):
    approx_q[:, action] = phi @ weights[:, action]  # Use the learned weights to get the approximated Q-values

print(f"Iterations: {iter}, MSE: {mse}, N. of Features {weights.size}")


# Reshape approx_q if needed for plotting
#approx_q = approx_q.reshape(-1, 5)  # Assuming Q has 5 actions
# needed for heatmaps
s_idx = np.ravel_multi_index(s.T, (9, 9))
unique_s_idx = np.unique(s_idx, return_index=True)[1]
n_actions = 5

# Increase the figure size (for example, width 15, height 8)
fig, axs = plt.subplots(2, n_actions, figsize=(10, 6))

for i, j in zip(range(n_actions), ["LEFT", "DOWN", "RIGHT", "UP", "STAY"]):
    axs[0][i].imshow(Q[unique_s_idx, i].reshape(9, 9), cmap='viridis')
    axs[1][i].imshow(approx_q[unique_s_idx, i].reshape(9, 9), cmap='viridis')
    
    # Set title with reduced font size
    axs[0][i].set_title(f"Q {j}", fontsize=10)
    axs[1][i].set_title(f"Approx. {name} \n(MSE {mse:.3f})", fontsize=10)
    
    # Adjust tick label font size
    axs[0][i].tick_params(axis='both', labelsize=10)
    axs[1][i].tick_params(axis='both', labelsize=10)

# Adjust spacing between subplots and layout
plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Increase space between plots if needed
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust tight layout margins

# Show the plot
plt.show()



#################### PART 5
# Discuss similarities and differences between SL regression and RL TD.
# - Discuss loss functions, techniques applicable to minimize it, and additional
#   challenges of RL.
# - What are the differences between "gradient descent" and "semi-gradient
#   descent" for TD?
# - Assume you'd have to learn the Q-function when actions are continuous.
#   How would you change your code?