import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

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
    




#################### PART 3
# Consider the Gridworld depicted below. The dataset below contains episodes
# collected using the optimal policy, and the heatmap below shows its V-function.
# - Consider the 5 FAs implemented earlier and discuss why each would be a
#   good/bad choice. Discuss each in at most two sentences.

# These data is a dictionary of (s, a, r, s', term, Q).
# - Run batch semi-gradient TD prediction with a FA of your choice (the one you
#   think would work best) to learn an approximation of the V-function.
#   Use gamma = 0.99. Increase the number of iterations, if you'd like.
#   Plot your result of the true V-function against your approximation using the
#   provided plotting function.

data = np.load("/media/adebayo/Windows/UofACanada/study/codes/CMPUT655_RL/a6_gridworld.npz")
s = data["s"]
a = data["a"]
r = data["r"]
s_next = data["s_next"]
Q = data["Q"]
V = data["Q"].max(-1)  # value of the greedy policy
term = data["term"]
n = s.shape[0]



state_size = 2
n_samples = 10
n_centers = 100
state = np.random.rand(n_samples, state_size)  # in [0, 1]

state_1_centers = np.linspace(-10, 10, n_centers)
state_2_centers = np.linspace(-10, 10, n_centers)
centers = np.array(
    np.meshgrid(state_1_centers, state_2_centers)
).reshape(state_size, -1).T  # makes a grid of uniformly spaced centers in the plane [-0.2, 1.2]^2

widths = 0.5  # Width for tile/coarse coding
sigmas = 0.5  # Sigma for RBFs

fig, axs = plt.subplots(1, 1)
axs.tricontourf(s[:, 0], s[:, 1], V, levels=100)
plt.show()

max_iter = 20000
gamma = 0.99
alpha = 1.0
thresh = 1e-8
offsets = [(-0.1, 0.0), (0.0, 0.1), (0.1, 0.0), (0.0, -0.1)]

# Pick one
# name, get_phi = "Poly", lambda state : poly_features(state, 2)
# name, get_phi = "RBFs", lambda state : rbf_features(state, centers, sigmas)
# name, get_phi = "Tiles", lambda state : tile_features(state, centers, widths, offsets)
name, get_phi = "coarse_features", lambda state : tile_features(state, centers, widths, offsets)
#name, get_phi = "Aggreg.", lambda state: aggregation_features(state, centers)

phi = get_phi(s)
phi_next = get_phi(s_next)
weights = np.zeros(phi.shape[-1])
pbar = tqdm(total=max_iter)


for iter in range(max_iter):
    # TD semi-gradient update
    td_target = r + (1-term)*gamma * V[s_next.argmax(axis=1)]
    Q_prediction = phi @ weights  # Predicted Q values
    td_error = td_target - Q_prediction  # Calculate TD Error

    # Calculate MSE
    mse = np.mean(td_error ** 2)  # Mean Squared Error

    # Update weights using TD error
    gradient = (phi.T @ td_error) / len(td_error)
   
    weights += alpha * gradient  # Use different alpha values for each method
    alpha = max(alpha - 1/max_iter, 0.001)

    pbar.set_description(f"TDE: {td_error.mean():.3f}, MSE: {mse:.3f}")
    pbar.update()
    if mse < thresh:
        break

print(f"Iterations: {iter}, MSE: {mse:.3f}, N. of Features: {len(weights)}")



# Visualization
fig, axs = plt.subplots(1, 2)
axs[0].tricontourf(s[:, 0], s[:, 1], V, levels=100)
td_prediction = phi @ weights  # Predicted Q values after training
axs[1].tricontourf(s[:, 0], s[:, 1], td_prediction, levels=100)
axs[0].set_title("True Function", fontsize=12)
axs[1].set_title(f"Approx. {name} (MSE {mse:.3f})", fontsize=12)  # Change to scientific notation if desired
axs[0].tick_params(axis='both', labelsize=16)
axs[1].tick_params(axis='both', labelsize=16)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()

