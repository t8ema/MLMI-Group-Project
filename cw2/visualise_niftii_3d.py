import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Print the current working directory (for debugging)
print("Current Working Directory:", os.getcwd())

# Set the path to the image and mask
image_path = "MLMI-Group-Project/cw2/data/001001_img.nii"
mask_path = "MLMI-Group-Project/cw2/data/001001_mask.nii"

# Load the image and mask using nibabel
img = nib.load(image_path)
mask = nib.load(mask_path)

# Extract the image data as NumPy arrays
img_data = img.get_fdata()
mask_data = mask.get_fdata()

# Print shapes to check if 3D
print("Image shape:", img_data.shape)
print("Mask shape:", mask_data.shape)

# Check if the data is 3D
if img_data.ndim == 3 and mask_data.ndim == 3:
    # Function to update the displayed slices
    def update_slice(val):
        z = int(slider.val)
        image_im.set_data(img_data[:, :, z].T)
        mask_im.set_data(mask_data[:, :, z].T)
        fig.canvas.draw_idle()

    # Create a figure with subplots for image and mask
    fig, (image_ax, mask_ax) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the initial middle slice
    z_middle = img_data.shape[2] // 2
    image_im = image_ax.imshow(img_data[:, :, z_middle].T, cmap="gray", origin="lower")
    image_ax.set_title("Image")
    image_ax.axis("off")
    cbar_image = fig.colorbar(image_im, ax=image_ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar_image.set_label("Intensity (Image)")

    mask_im = mask_ax.imshow(mask_data[:, :, z_middle].T, cmap="gray", origin="lower")
    mask_ax.set_title("Mask")
    mask_ax.axis("off")
    cbar_mask = fig.colorbar(mask_im, ax=mask_ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar_mask.set_label("Intensity (Mask)")

    # Add a slider to navigate through slices
    ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03], facecolor="lightgray")
    slider = Slider(ax_slider, "Slice", 0, img_data.shape[2] - 1, valinit=z_middle, valfmt="%d")

    # Update the plots when the slider value changes
    slider.on_changed(update_slice)

    plt.tight_layout()
    plt.show()
else:
    print("The image or mask is not 3D!")
  
