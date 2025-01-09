import os
import numpy as np
import tensorflow as tf


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
    # Define paths to models and validation folder
    model_save_paths = [
        './MLMI-Group-Project/cw2/saved_models/40_220.tf',
        './MLMI-Group-Project/cw2/saved_models/42_210.tf',
        './MLMI-Group-Project/cw2/saved_models/45_285.tf',  # Add more model paths here
    ]
    path_to_val_folder = './MLMI-Group-Project/cw2/val'  # Validation folder path
    size_minibatch = 6  # Minibatch size
    binarize_mask = True
    binary_threshold = 0.5

    # Load models
    loaded_models = [tf.saved_model.load(path) for path in model_save_paths]

    dice_scores_per_model = [[] for _ in loaded_models]  # Store Dice scores for each model
    dice_scores_ensemble = []  # Store Dice scores for ensemble predictions

    # Get the list of all image files in the validation folder
    image_files = sorted([f for f in os.listdir(path_to_val_folder) if f.startswith("image_val") and f.endswith(".npy")])
    num_images = len(image_files)

    for i in range(num_images):  # Loop through all validation images dynamically
        # Generate file paths
        image_file = os.path.join(path_to_val_folder, f"image_val{i:03d}.npy")
        truth_mask_file = os.path.join(path_to_val_folder, f"label_val{i:03d}.npy")

        # Load and preprocess the image
        image = np.load(image_file)
        gray_chan_image = np.expand_dims(image, axis=-1)
        batch_dim_image = np.expand_dims(gray_chan_image, axis=0)
        stacked_image = np.tile(batch_dim_image, (size_minibatch, 1, 1, 1, 1))  # Duplicate the image to fit batch size

        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(stacked_image, dtype=tf.float32)

        # Get predictions from each model
        predictions = []
        for model in loaded_models:
            pred = model(input_tensor)[0, :, :, :, 0].numpy()
            if binarize_mask:
                pred = (pred > binary_threshold).astype(np.uint8)
            predictions.append(pred)

        # Calculate Dice coefficients for each model
        truth_mask = np.load(truth_mask_file)
        for model_idx, pred in enumerate(predictions):
            dice = calculate_dice_coefficient(pred, truth_mask)
            dice_scores_per_model[model_idx].append(dice)

        # Average the predictions for ensemble
        avg_prediction = np.mean(predictions, axis=0)
        if binarize_mask:
            avg_prediction = (avg_prediction > binary_threshold).astype(np.uint8)

        # Calculate Dice coefficient for the ensemble prediction
        dice_ensemble = calculate_dice_coefficient(avg_prediction, truth_mask)
        dice_scores_ensemble.append(dice_ensemble)

    # Calculate the average Dice coefficients
    average_dice_per_model = [np.mean(scores) for scores in dice_scores_per_model]
    average_dice_ensemble = np.mean(dice_scores_ensemble)

    # Display results
    for model_idx, avg_dice in enumerate(average_dice_per_model):
        print(f"Average Dice coefficient for model {model_idx + 1}: {avg_dice:.4f}")
    print(f"Average Dice coefficient for ensemble prediction: {average_dice_ensemble:.4f}")
