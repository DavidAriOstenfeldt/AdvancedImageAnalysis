# %%
import skimage.io as io
import matplotlib.pyplot as plt
import scipy
import numpy as np
import os
# %%
# Load data
data_path = '../../Data/Chapter1/fuel_cells/'

# Load all images in the folder
img_list = []
for file in os.listdir(data_path):
    if file.endswith('.tif'):
        img = io.imread(data_path + file)
        img_list.append(img)

# %%
# Visualize the images
fig, axes = plt.subplots(1, len(img_list), figsize=(12, 4))
ax = axes.ravel()
for i in range(len(img_list)):
    ax[i].imshow(img_list[i], cmap='gray')
    ax[i].set_title('Image ' + str(i + 1))
    
plt.show()

# %%
# Vectorize the images, and calculate length of segmentation boundary
boundary_length = []
for img in img_list:
    print(np.sum(img[:-1] != img[1:]))


# %%
# Create function to calculate length of segmentation boundary
def calculate_boundary_length(img):
    boundary_length = np.sum(img[:-1] != img[1:])
    return boundary_length

# Calculate boundary length for each image
boundary_length = []
for img in img_list:
    boundary_length.append(calculate_boundary_length(img))

print(boundary_length)
# %%
