# Allows you to view the results of the training
# Ensure you match the image_test with the correct corresponding label manually

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



# Parameters
image_file = "MLMI-Group-Project/cw2/processed_data/image_test21.npy"  # Directory where images are stored
truth_mask_file = "MLMI-Group-Project/cw2/processed_data/label_test21.npy"  # Directory where masks are stored
predicted_mask_file = "MLMI-Group-Project/cw2/results/label_test21_step000128-tf.npy"  # Directory where result masks are stored



def visualize_data(image_data, predicted_mask_data=None, truth_mask_data=None):
    """
    Visualize 3D image data with three subplots: CT scan, ground truth mask, and predicted mask.

    Parameters:
        image_data (numpy.ndarray): The 3D image data (shape: slices x height x width).
        predicted_mask_data (numpy.ndarray, optional): The 3D predicted mask data (same shape as image_data).
        truth_mask_data (numpy.ndarray, optional): The 3D ground truth mask data (same shape as image_data).
    """
    # Check if the dimensions match
    if (predicted_mask_data is not None and image_data.shape != predicted_mask_data.shape) or \
       (truth_mask_data is not None and image_data.shape != truth_mask_data.shape):
        raise ValueError("Image data, predicted mask data, and truth mask data must have the same shape.")

    # Initialize the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)  # Make space for the slider

    # Initial slice to display
    initial_slice = 0
    img = image_data[initial_slice, :, :]
    predicted_mask = predicted_mask_data[initial_slice, :, :] if predicted_mask_data is not None else None
    truth_mask = truth_mask_data[initial_slice, :, :] if truth_mask_data is not None else None

    # Subplot 1: CT scan
    ct_display = axes[0].imshow(img, cmap='gray')
    axes[0].set_title("CT Scan")
    axes[0].axis('off')

    # Subplot 2: Ground Truth Mask
    if truth_mask_data is not None:
        truth_display = axes[1].imshow(truth_mask, cmap='gray', alpha=1.0)
    else:
        truth_display = axes[1].imshow(np.zeros_like(img), cmap='gray', alpha=1.0)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    # Subplot 3: Predicted Mask
    if predicted_mask_data is not None:
        predicted_display = axes[2].imshow(predicted_mask, cmap='gray', alpha=1.0)
    else:
        predicted_display = axes[2].imshow(np.zeros_like(img), cmap='gray', alpha=1.0)
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    # Slider widget
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position of the slider
    slider = Slider(ax_slider, "Slice", 0, image_data.shape[0] - 1, valinit=initial_slice, valstep=1)

    # Update function for the slider
    def update(val):
        slice_idx = int(slider.val)  # Get the current slider value
        img = image_data[slice_idx, :, :]
        predicted_mask = predicted_mask_data[slice_idx, :, :] if predicted_mask_data is not None else None
        truth_mask = truth_mask_data[slice_idx, :, :] if truth_mask_data is not None else None

        # Update CT scan
        ct_display.set_data(img)

        # Update ground truth mask
        if truth_mask_data is not None:
            truth_display.set_data(truth_mask)

        # Update predicted mask
        if predicted_mask_data is not None:
            predicted_display.set_data(predicted_mask)

        fig.canvas.draw_idle()

    # Connect the update function to the slider
    slider.on_changed(update)

    plt.show()

# Example usage
if __name__ == "__main__":
    # Load image and mask data
    image_data = np.load(image_file)
    image_data = image_data[::2, ::2, ::2]
    predicted_mask_data = np.load(predicted_mask_file)
    truth_mask_data = np.load(truth_mask_file)
    truth_mask_data = truth_mask_data[::2, ::2, ::2]

    print(image_data.shape)
    print(predicted_mask_data.shape)
    print(truth_mask_data.shape)

    # Call the visualization function
    visualize_data(image_data, predicted_mask_data, truth_mask_data)
  
