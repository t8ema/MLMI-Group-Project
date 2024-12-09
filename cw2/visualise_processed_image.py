# This script allows you to visualise a processed image for checking

image_index = 0  # Index to select an image



import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



# Parameters
image_dir = "MLMI-Group-Project/cw2/processed_data/image"  # Directory where images are stored
mask_dir = "MLMI-Group-Project/cw2/processed_data/mask"  # Directory where masks are stored

# Get a list of all image and mask files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".npy")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy")])

# Ensure there is a corresponding mask for each image
assert len(image_files) == len(mask_files), "The number of image and mask files do not match!"



# Load image and mask data
image_file = image_files[image_index]
mask_file = mask_files[image_index]

image_data = np.load(os.path.join(image_dir, image_file))
mask_data = np.load(os.path.join(mask_dir, mask_file))



# Check if the loaded data is 3D
assert image_data.ndim == 3, "Image data is not 3D!"
assert mask_data.ndim == 3, "Mask data is not 3D!"



# Initialize the figure and axis for plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Initial slice (middle slice)
initial_slice = image_data.shape[2] // 2

# Plot the image and mask
img_plot = ax[0].imshow(image_data[:, :, initial_slice], cmap="gray", origin="lower")
mask_plot = ax[1].imshow(mask_data[:, :, initial_slice], cmap="gray", origin="lower")

# Titles for each plot
ax[0].set_title(f"Image: {image_file}")
ax[1].set_title(f"Mask: {mask_file}")
ax[0].axis("off")
ax[1].axis("off")

# Colorbars
fig.colorbar(img_plot, ax=ax[0])
fig.colorbar(mask_plot, ax=ax[1])

# Slider for selecting slices
ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor="lightgoldenrodyellow")
slice_slider = Slider(
    ax_slider,
    "Slice",
    0,
    image_data.shape[2] - 1,
    valinit=initial_slice,
    valstep=1
)

# Update function for the slider
def update(val):
    slice_idx = int(slice_slider.val)
    img_plot.set_data(image_data[:, :, slice_idx])
    mask_plot.set_data(mask_data[:, :, slice_idx])
    fig.canvas.draw_idle()

# Connect the slider to the update function
slice_slider.on_changed(update)

# Show the plot
plt.tight_layout()
plt.show()
