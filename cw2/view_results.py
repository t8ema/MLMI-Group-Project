# Allows you to view the results slightly more intuitively than visualise.py
# Ensure you match the image_test with the correct corresponding label manually

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



# Parameters
image_file = "MLMI-Group-Project/cw2/processed_data/image_test29.npy"  # Point to the image file
mask_file = "MLMI-Group-Project/cw2/results/label_test29_step000016-tf.npy"  # Point to the corresponding mask file



def visualize_data(image_data, mask_data=None):
    """
    Visualize 3D image data with three subplots: CT scan, mask, and CT scan with the mask overlay.

    Parameters:
        image_data (numpy.ndarray): The 3D image data (shape: slices x height x width).
        mask_data (numpy.ndarray, optional): The 3D mask data (same shape as image_data).
    """
    # Check if the dimensions match
    if mask_data is not None and image_data.shape != mask_data.shape:
        raise ValueError("Image data and mask data must have the same shape.")

    # Initialize the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)  # Make space for the slider

    # Initial slice to display
    initial_slice = 0
    img = image_data[initial_slice, :, :]
    mask = mask_data[initial_slice, :, :] if mask_data is not None else None

    # Subplot 1: CT scan
    ct_display = axes[0].imshow(img, cmap='gray')
    axes[0].set_title("CT Scan")
    axes[0].axis('off')

    # Subplot 2: Mask
    if mask_data is not None:
        mask_display = axes[1].imshow(mask, cmap='gray', alpha=1.0)
    else:
        mask_display = axes[1].imshow(np.zeros_like(img), cmap='gray', alpha=1.0)
    axes[1].set_title("Mask")
    axes[1].axis('off')

    # Subplot 3: CT scan with mask overlay
    overlay = np.ma.masked_where(mask == 0, mask) if mask_data is not None else None
    combined_display = axes[2].imshow(img, cmap='gray')
    if mask_data is not None:
        overlay_display = axes[2].imshow(overlay, cmap='gray', alpha=0.9)
    axes[2].set_title("CT Scan with Mask")
    axes[2].axis('off')

    # Slider widget
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position of the slider
    slider = Slider(ax_slider, "Slice", 0, image_data.shape[0] - 1, valinit=initial_slice, valstep=1)

    # Update function for the slider
    def update(val):
        slice_idx = int(slider.val)  # Get the current slider value
        img = image_data[slice_idx, :, :]
        mask = mask_data[slice_idx, :, :] if mask_data is not None else None

        # Update CT scan
        ct_display.set_data(img)

        # Update mask
        if mask_data is not None:
            mask_display.set_data(mask)

        # Update CT scan with mask overlay
        if mask_data is not None:
            overlay = np.ma.masked_where(mask == 0, mask)
            combined_display.set_data(img)
            overlay_display.set_data(overlay)
        fig.canvas.draw_idle()

    # Connect the update function to the slider
    slider.on_changed(update)

    plt.show()



# Example usage
if __name__ == "__main__":
    # Load example image and mask data (replace with your data paths)
    image_data = np.load(image_file)
    image_data = image_data[::2, ::2, ::2]
    mask_data = np.load(mask_file)

    # Print shapes for checking. Should be downsampled by half
  
    print(image_data.shape)
    print(mask_data.shape)

    # Call the visualization function
    visualize_data(image_data, mask_data)
  
