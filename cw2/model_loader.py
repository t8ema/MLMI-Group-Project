# This script allows you to use a saved model to make a prediction on an arbitary file - i.e. any file you choose



import os
import random

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



model_save_path = './MLMI-Group-Project/cw2/saved_models/residual_unet_step_124.tf' # Path to the model you want to load

path_to_image = './MLMI-Group-Project/cw2/processed_data/image_test00.npy' # Path to image you want to predict on
truth_mask_file = "./MLMI-Group-Project/cw2/processed_data/label_test00.npy"  # Path to relevant truth mask file

binarize_mask = True  # When True, make predicted mask binary (predicted values > binary_threshold become 1, and lower values become 0)
binary_threshold = 0.5  # Threshold for binary masks - the default standard is 0.5



# Load image and process into format accepted by the model (downsample, grayscale channel, batch size dimension)
image = np.load(path_to_image)  # Load the image
print('Loaded image shape: ', image.shape)

downsampled_image = image[::2, ::2, ::2]  # Downsample the image
print('Downsampled image shape: ', downsampled_image.shape)

gray_chan_image = np.expand_dims(downsampled_image, axis=-1)  # Add the grayscale channel
print('Grayscale channel image shape: ', gray_chan_image.shape)

batch_dim_image = np.expand_dims(gray_chan_image, axis=0)  # Add the batch dimension
print('Batch dimension channel image shape: ', batch_dim_image.shape)

stacked_image = np.tile(batch_dim_image, (4, 1, 1, 1, 1))  # Duplicate the image to fit batch size of 4
print('Stacked (batch size) image shape: ', stacked_image.shape)

# Convert to TensorFlow tensor
input_tensor = tf.convert_to_tensor(stacked_image, dtype=tf.float32)
print(input_tensor.shape)



# Choose the loaded model
loaded_model = tf.saved_model.load(model_save_path)

# Call the loaded model
pred = loaded_model(input_tensor)

print('Shape of predicted image: ', pred.shape)



print('-------------------------------------------------')



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




if __name__ == "__main__":
    # Load the image
    image_data = downsampled_image
    
    # Load the predicted mask
    predicted_mask_data = pred[0, :, :, :, 0]
    
    if binarize_mask == True:
        # Make the mask binary
        predicted_mask_data = tf.where(predicted_mask_data > binary_threshold, 1, 0)
    
    # Convert to NumPy array
    predicted_mask_data = predicted_mask_data.numpy()
    
    # Load the truth mask
    truth_mask_data = np.load(truth_mask_file)
    truth_mask_data = truth_mask_data[::2, ::2, ::2]

    print('Image shape for plotting: ', image_data.shape)
    print('Predicted mask shape for plotting: ', predicted_mask_data.shape)
    print('Truth mask shape for plotting: ', truth_mask_data.shape)

    # Call the visualization function
    visualize_data(image_data, predicted_mask_data, truth_mask_data)
