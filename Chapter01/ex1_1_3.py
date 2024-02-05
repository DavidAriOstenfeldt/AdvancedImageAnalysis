# %%
import matplotlib.pyplot as plt
import scipy
import numpy as np
# %%
# Load data
data_path = '../../Data/Chapter1/curves/'

dino_noisy = np.loadtxt(data_path + 'dino_noisy.txt')
dino = np.loadtxt(data_path + 'dino.txt')
hand_noisy = np.loadtxt(data_path + 'hand_noisy.txt')
hand = np.loadtxt(data_path + 'hand.txt')

# %%
# Implement curve smoothing
def smooth_curve(X, LAMBDA=0.5):
    N = len(X)
    first_row = np.array([-2, 1] + [0] * (N - 3) + [1])
    L = scipy.linalg.circulant(first_row)
    X_new = (np.eye(N) + LAMBDA * L) @ X
    return X_new

X_new = smooth_curve(dino_noisy)

# Plot original and smoothed curves
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
marker_size = 5
ax[0].plot(dino_noisy[:, 0], dino_noisy[:, 1], '.-c', ms=marker_size)
ax[0].set_title('Noisy Curve')
ax[1].plot(X_new[:, 0], X_new[:, 1], '.-c', ms=marker_size)
ax[1].set_title('Smoothed Curve')
plt.show()


# %%
# Iteratively smooth the curve
from IPython.display import display, clear_output
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
marker_size = 5
X_new = dino_noisy
for i in range(10):
    X_new = smooth_curve(X_new)
    ax[1].clear()
    ax[0].plot(dino_noisy[:, 0], dino_noisy[:, 1], '.-c', ms=marker_size)
    ax[0].set_title('Noisy Curve')
    ax[1].plot(X_new[:, 0], X_new[:, 1], '.-c', ms=marker_size)
    ax[1].set_title('Smoothed Curve')
    display(fig)
    clear_output(wait=True)

plt.close()
    
# %%
# implement curve iterative curve smoothing function
def iterative_smooth_curve(X, n_iter=10, LAMBDA=0.5):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()
    marker_size = 5
    X_new = X
    for i in range(n_iter):
        X_new = smooth_curve(X_new, LAMBDA)
        ax[1].clear()
        ax[0].plot(X[:, 0], X[:, 1], '.-c', ms=marker_size)
        ax[0].set_title('Noisy Curve')
        ax[1].plot(X_new[:, 0], X_new[:, 1], '.-c', ms=marker_size)
        ax[1].set_title('Smoothed Curve')
        display(fig)
        clear_output(wait=True)
    plt.close()
# %%
iterative_smooth_curve(dino_noisy, 10, 0.5)
# %%
# Implement implicit smoothing
def implicit_smooth_curve(X, LAMBDA=0.5):
    N = len(X)
    first_row = np.array([-2, 1] + [0] * (N - 3) + [1])
    L = scipy.linalg.circulant(first_row)
    X_new = np.linalg.inv(np.eye(N) - LAMBDA*L) @ X
    return X_new

X_new = implicit_smooth_curve(dino_noisy)
# Plot original and smoothed curves
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
marker_size = 5
ax[0].plot(dino_noisy[:, 0], dino_noisy[:, 1], '.-c', ms=marker_size)
ax[0].set_title('Noisy Curve')
ax[1].plot(X_new[:, 0], X_new[:, 1], '.-c', ms=marker_size)
ax[1].set_title('Smoothed Curve')
plt.show()

# %%
# Implement implicit curve smoothing with extended kernel
def implicit_smooth_curve_extended(X, alpha=.5, beta=.5):
    N = len(X)
    first_row = np.array([-2, 1] + [0] * (N - 3) + [1])
    A = scipy.linalg.circulant(first_row)
    first_row = np.array([-1, 4, -6, 4, -1] + [0] * (N - 5))
    B = np.array(first_row)
    X_new = np.linalg.inv(np.eye(N) - alpha * A - beta * B) @ X
    return X_new

X_new = implicit_smooth_curve_extended(dino_noisy)
# Plot original and smoothed curves
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
marker_size = 5
ax[0].plot(dino_noisy[:, 0], dino_noisy[:, 1], '.-c', ms=marker_size)
ax[0].set_title('Noisy Curve')
ax[1].plot(X_new[:, 0], X_new[:, 1], '.-c', ms=marker_size)
ax[1].set_title('Smoothed Curve')
plt.show()


# %%
# Implement function that gives smoothing kernel given N, alpha and beta
def create_smoothing_kernel(N, alpha, beta):
    first_row = np.array([-2, 1] + [0] * (N - 3) + [1])
    A = scipy.linalg.circulant(first_row)
    first_row = np.array([-1, 4, -6, 4, -1] + [0] * (N - 5))
    B = np.array(first_row)
    kernel = np.linalg.inv(np.eye(N) - alpha * A - beta * B)
    return kernel
