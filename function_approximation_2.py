import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures


np.set_printoptions(precision=3, suppress=True)

#env = gym.envs.make("a6_gridworld")
# Notation for array sizes:
# - S: state dimensionality
# - D: features dimensionality
# - N: number of samples
#
# N is always the first dimension, meaning that states come in arrays of shape (N, S)
# and features in arrays of shape (N, D).
# We recommend to implement the functions below assuming that the input has
# always shape (N, S) and the output (N, D), even when N = 1.



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

# Generate example data
state_size = 2
n_samples = 10
n_centers = 100
state = np.random.rand(n_samples, state_size)  # in [0, 1]

state_1_centers = np.linspace(-0.2, 1.2, n_centers)
state_2_centers = np.linspace(-0.2, 1.2, n_centers)
centers = np.array(np.meshgrid(state_1_centers, state_2_centers)).reshape(state_size, -1).T
sigmas = 0.2
widths = 0.2
offsets = [(-0.1, 0.0), (0.0, 0.1), (0.1, 0.0), (0.0, -0.1)]

# Example feature computations
poly = poly_features(state, 2)
aggr = aggregation_features(state, centers)
rbf = rbf_features(state, centers, sigmas)
tile_one = tile_features(state, centers, widths)
tile_multi = tile_features(state, centers, widths, offsets)
coarse_one = coarse_features(state, centers, widths)
coarse_multi = coarse_features(state, centers, widths, offsets)

# Visualize
fig, axs = plt.subplots(1, 6, figsize=(18, 6))
extent = [state_1_centers[0], state_1_centers[-1], state_2_centers[0], state_2_centers[-1]]
axs[0].imshow(rbf[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[1].imshow(tile_one[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[2].imshow(tile_multi[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[3].imshow(coarse_one[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[4].imshow(coarse_multi[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[5].imshow(aggr[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
titles = ["RBFs", "Tile (1 Tiling)", "Tile (4 Tilings)", "Coarse (1 Field)", "Coarse (4 Fields)", "Aggreg."]
for ax, title in zip(axs, titles):
    ax.plot(state[0][0], state[0][1], marker="+", markersize=18, color="red")
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
plt.suptitle(f"State {state[0]}", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



