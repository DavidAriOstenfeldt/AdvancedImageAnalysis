# %%
import skimage.io as io
import matplotlib.pyplot as plt
import scipy
import numpy as np

# %%
# Load image
data_path = '../../Data/Chapter1/'
img = io.imread(data_path + 'fibres_xcth.png')


sigma = 4.5
kernel_radius = 4 * sigma

# Create Gaussian Kernel
def create_gaussian_kernel(x, sigma=4.5):
    g = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-(x**2 / (2 * sigma**2)))
    return g

x = np.array(np.arange(-kernel_radius, kernel_radius + 1, 1, dtype=int))
g = create_gaussian_kernel(x, sigma)


# %%
# Plot x against g to verify correctness
plt.plot(x, g)
plt.title('Gaussian Kernel')
plt.xlabel('x')
plt.ylabel('g')
plt.show()

# %%
# Create derived Gaussian Kernel
def create_derived_gaussian_kernel(x, sigma=4.5):
    dg = -(x / (sigma**3 * np.sqrt(2 * np.pi))) * np.exp(-(x**2 / (2 * sigma**2)))
    return dg

dg = create_derived_gaussian_kernel(x, sigma)

# Plot x against dg to verify correctness
plt.plot(x, dg)
plt.title('Gaussian Kernel')
plt.xlabel('x')
plt.ylabel('dg')
plt.show()

# %%
# Create 2D Gaussian Kernel by getting outer product of two 1D Gaussian Kernels
g2d = np.outer(g, g)
img_convolved_2d = scipy.ndimage.convolve(img, g2d)

# Plot original and convolved images in a 1x2 grid
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(img_convolved_2d, cmap='gray')
ax[1].set_title('Convolved Image')
plt.show()

# %%
# Verify that the 2D convolution is the same as two 1D orthogonal convolutions
img_convolved_1d = scipy.ndimage.convolve(img.flatten(), g)
img_convolved_1d = scipy.ndimage.convolve(img_convolved_1d, g.T)
img_convolved_1d = img_convolved_1d.reshape(img.shape)

# Plot original and convolved images in a 1x2 grid
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(img_convolved_1d, cmap='gray')
ax[1].set_title('Convolved Image')
plt.show()

# Plot the difference between the two convolved images
img_convolved_1d = img_convolved_1d.astype(int)
img_convolved_2d = img_convolved_2d.astype(int)
diff_x = img_convolved_2d[1:,:] - img_convolved_1d[:-1,:]
plt.imshow(diff_x, cmap='bwr')


# %%
# Verify the derivative
img_convolved = scipy.ndimage.convolve(img.flatten(), g)
img_convolved = scipy.ndimage.convolve(img_convolved, [0.5, 0, -0.5])
img_convolved = img_convolved.reshape(img.shape)

img_convolved_dg = scipy.ndimage.convolve(img.flatten(), dg)
img_convolved_dg = img_convolved_dg.reshape(img.shape)

# Plot the original and convolved images in a 1x3 grid
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ax = axes.ravel()
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(img_convolved, cmap='gray')
ax[1].set_title('Convolved Image (Gaussian)')
ax[2].imshow(img_convolved_dg, cmap='gray')
ax[2].set_title('Convolved Image (Derivative)')
plt.show()

# Plot the difference between the two convolved images
img_convolved = img_convolved.astype(int)
img_convolved_dg = img_convolved_dg.astype(int)
diff_x = img_convolved[1:,:] - img_convolved_dg[:-1,:]
plt.imshow(diff_x, cmap='bwr')

# %%
# Verify that a convolution with a Gaussian of t=20 is equal to ten convolutions with t=2
sigma = np.sqrt(20)
g = create_gaussian_kernel(x, sigma)
img_convolved_20 = scipy.ndimage.convolve(img.flatten(), g)
img_convolved_20 = img_convolved_20.reshape(img.shape)

sigma = np.sqrt(2)
g = create_gaussian_kernel(x, sigma)
img_convolved_2 = scipy.ndimage.convolve(img.flatten(), g)
for i in range(9):
    img_convolved_2 = scipy.ndimage.convolve(img_convolved_2, g)
img_convolved_2 = img_convolved_2.reshape(img.shape)

# Plot the original and convolved images in a 1x3 grid
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ax = axes.ravel()
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(img_convolved_20, cmap='gray')
ax[1].set_title('Convolved Image (t=20)')
ax[2].imshow(img_convolved_2, cmap='gray')
ax[2].set_title('Convolved Image (t=2) x 10')
plt.show()

# Plot the difference between the two convolved images
img_convolved_20 = img_convolved_20.astype(int)
img_convolved_2 = img_convolved_2.astype(int)
diff_x = img_convolved_20[1:,:] - img_convolved_2[:-1,:]
plt.imshow(diff_x, cmap='bwr')


# %%
# Compare gaussian with scipy gaussian
g = create_gaussian_kernel(x, 4.5)
img_convolved = scipy.ndimage.convolve(img, np.outer(g, g))
s_g = scipy.ndimage.gaussian_filter(img, 4.5)

# Plot the difference between the two convolved images
diff_x = img_convolved[:1, :] - s_g[:-1, :]
plt.imshow(diff_x, cmap='bwr')

# %%
# Answer Quiz Question
# Load noisy_number_2023.png
img = io.imread(data_path + 'noisy_number_2023.png')

# Create Gaussian Kernel
sigma = 4.5
kernel_radius = 4 * sigma
x = np.array(np.arange(-kernel_radius, kernel_radius + 1, 1, dtype=int))
g = create_gaussian_kernel(x, sigma)

# Convolve image with kernel
img_convolved = scipy.ndimage.convolve(img, np.outer(g, g))

# Plot original and convolved images in a 1x2 grid
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(img_convolved, cmap='gray')
ax[1].set_title('Convolved Image')
plt.show()


# %%
