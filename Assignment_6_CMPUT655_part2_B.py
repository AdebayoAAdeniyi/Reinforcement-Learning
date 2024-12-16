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
    

# Training setup
x = np.linspace(-10, 10, 100)
y = np.zeros(x.shape)
y[0:10] = x[0:10]**3 / 3.0
y[10:20] = np.exp(x[25:35])
y[20:30] = -x[0:10]**3 / 2.0
y[30:60] = 100.0
y[60:70] = 0.0
y[70:100] = np.cos(x[70:100]) * 100.0
fig, axs = plt.subplots(1, 1)
axs.plot(x, y)
plt.show()
max_iter = 50000
thresh = 1e-8
alpha = 1.0

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

# Loop through each feature approach
for name, get_phi in zip(
    ["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."],
    [
        lambda state: poly_features(state, 3),
        lambda state: rbf_features(state, centers, sigmas),
        lambda state: tile_features(state, centers, widths),
        lambda state: coarse_features(state, centers, widths),
        lambda state: aggregation_features(state, centers),
    ]
):
    # Get the feature matrix
    phi = get_phi(x[..., None])

    # Select the third feature only for poly_features
    if name == "Poly":
        phi_selected = phi  # Select the 4th column (feature 3, index 3)
    else:
        phi_selected = phi  # Use all features for other methods

    # Initialize weights for the selected feature
    weights = np.zeros(phi_selected.shape[-1])

    # Define y_hat function using the selected features
    y_hat = lambda x, weights: (phi_selected @ weights)  # Matrix multiplication

    # Run gradient descent
    pbar = tqdm(total=max_iter)
    for iter in range(max_iter):
        if name == "Poly":
            alpha = alpha/1000
        else:
            alpha = alpha
        # Compute the model output
        y_pred = y_hat(x[..., None], weights)

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((y_pred - y) ** 2)

        # Check for convergence
        if mse < thresh:
            break

        # Compute the residuals and gradient
        residuals = y_pred - y
        gradient = (1 / len(y)) * (phi_selected.T @ residuals)  # Gradient of MSE

        # Update the weights using gradient descent
        weights -= alpha * gradient  
        alpha = max(alpha - 1/max_iter, 0.001)

        # Update progress bar
        pbar.set_description(f"MSE: {mse:.6f}")
        pbar.update()

    pbar.close()

    
    # After training, compute the final predictions for plotting
    y_hat_values = phi_selected @ weights
    # Plot true function and the approximation
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x, y)
    axs[1].plot(x, y_hat_values)
    axs[0].set_title("True Function")
    axs[1].set_title(f"Approximation: {name} (MSE {mse:.3f})")
    plt.show()
    axs[0].tick_params(axis='both', labelsize=12)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[0].legend(fontsize=12)
    axs[1].legend(fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


