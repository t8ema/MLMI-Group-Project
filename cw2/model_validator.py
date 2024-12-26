# This script validates the model by producing an average dice over the whole dataset



import os
import random

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



model_save_path = './MLMI-Group-Project/cw2/saved_models/residual_unet_step_4.tf' # Path to the model you want to load

path_to_val_folder = './MLMI-Group-Project/cw2/val' # Path to validation folder
# images and labels have names image_val000.npy up to image_val099.npy, label_val000.npy to label_val099.npy

binarize_mask = True  # When True, make predicted mask binary (predicted values > binary_threshold become 1, and lower values become 0)
binary_threshold = 0.5  # Threshold for binary masks - the default standard is 0.5







# Choose the loaded model
loaded_model = tf.saved_model.load(model_save_path)



print('-------------------------------------------------')




def calculate_dice_coefficient(predicted, truth):
    """
    Calculate the Dice coefficient for 3D binary masks.

    Parameters:
        predicted (numpy.ndarray): Predicted binary mask (3D).
        truth (numpy.ndarray): Ground truth binary mask (3D).

    Returns:
        float: Dice coefficient.
    """
    intersection = np.sum(predicted * truth)
    volume_sum = np.sum(predicted) + np.sum(truth)
    if volume_sum == 0:
        return 1.0  # Perfect score if both masks are empty
    return (2.0 * intersection) / volume_sum


if __name__ == "__main__":
    dice_scores = []

    for i in range(100):  # Loop through all validation images
        # Generate file paths
        image_file = os.path.join(path_to_val_folder, f"image_val{i:03d}.npy")
        truth_mask_file = os.path.join(path_to_val_folder, f"label_val{i:03d}.npy")

        # Load and preprocess the image
        image = np.load(image_file)
        #downsampled_image = image[::2, ::2, ::2]
        #gray_chan_image = np.expand_dims(downsampled_image, axis=-1)
        gray_chan_image = np.expand_dims(image, axis=-1)
        batch_dim_image = np.expand_dims(gray_chan_image, axis=0)
        stacked_image = np.tile(batch_dim_image, (4, 1, 1, 1, 1))  # Duplicate the image to fit batch size of 4

        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(stacked_image, dtype=tf.float32)

        # Make predictions
        pred = loaded_model(input_tensor)[0, :, :, :, 0].numpy()

        if binarize_mask:
            pred = (pred > binary_threshold).astype(np.uint8)

        # Load and downsample the ground truth mask
        truth_mask = np.load(truth_mask_file)
        print('Shape of truth mask: ', truth_mask.shape)
        #truth_mask = truth_mask[::2, ::2, ::2]

        # Calculate the Dice coefficient
        dice = calculate_dice_coefficient(pred, truth_mask)
        dice_scores.append(dice)

        #print(f"Processed image {i:03d}, Dice coefficient: {dice:.4f}")

    # Calculate the average Dice coefficient
    average_dice = np.mean(dice_scores)
    print(f"Average Dice coefficient over the validation set: {average_dice:.4f}")
