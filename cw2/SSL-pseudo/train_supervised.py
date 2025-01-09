# Train supervised model for semi-supervised experiment

import os
import random
import shutil

import tensorflow as tf
import numpy as np



os.environ["CUDA_VISIBLE_DEVICES"]="0"
path_to_data = './MLMI-Group-Project/cw2/train'
path_to_test = './MLMI-Group-Project/cw2/test'
SAVE_PATH = './MLMI-Group-Project/cw2/saved_models'



# Set seeds for reproducibility
SEED = 40  # Seed of 40
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Enable deterministic operations in TensorFlow
os.environ["TF_DETERMINISTIC_OPS"] = "1"



# Define dice coefficient calculator
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

# Define data tester
def test_model(model, path_to_test_folder, binarize_mask=True, binary_threshold=0.5, minibatch_size=8):
    """
    Test the model on unseen data and calculate the average Dice coefficient.

    Parameters:
        model_path (str): Path to the saved model.
        path_to_test_folder (str): Path to the folder containing test images and labels.
        binarize_mask (bool): Whether to binarize the predicted masks.
        binary_threshold (float): Threshold for binarization.
        batch_size (int): Number of images in a batch.

    Returns:
        float: Average Dice coefficient over the test set.
    """
    # Load the model
    #model = tf.saved_model.load(model_path)

    # Get the list of all image files in the test folder
    image_files = sorted([f for f in os.listdir(path_to_test_folder) if f.startswith("image_test") and f.endswith(".npy")])
    num_images = len(image_files)

    dice_scores = []

    # Process test data in batches
    for start_idx in range(0, num_images, minibatch_size):
        end_idx = min(start_idx + minibatch_size, num_images)
        current_batch_size = end_idx - start_idx

        # Prepare batch data
        batch_images = []
        batch_truths = []

        for i in range(start_idx, end_idx):
            # Load the image
            image_file = os.path.join(path_to_test_folder, f"image_test{i:03d}.npy")
            truth_mask_file = os.path.join(path_to_test_folder, f"label_test{i:03d}.npy")

            image = np.load(image_file)
            truth_mask = np.load(truth_mask_file)

            gray_chan_image = np.expand_dims(image, axis=-1)
            batch_images.append(gray_chan_image)
            batch_truths.append(truth_mask)

        # Stack images into a single batch
        batch_images = np.stack(batch_images, axis=0)
        batch_truths = np.stack(batch_truths, axis=0)

        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(batch_images, dtype=tf.float32)

        # If the number of images is not divisible by the minibatch size, the final test minibatch will be 
        # unequal to the minibatch size. Hence, we make sure here that the first dimension of the tensor is
        # the correct size (minibatch size). If not, we go back to the start of the loop
        if input_tensor.shape[0] != minibatch_size:
            break
        
        # Make predictions
        predictions = model(input_tensor)

        for j in range(current_batch_size):
            pred = predictions[j, :, :, :, 0].numpy()

            if binarize_mask:
                pred = (pred > binary_threshold).astype(np.uint8)

            # Calculate the Dice coefficient
            dice = calculate_dice_coefficient(pred, batch_truths[j])
            dice_scores.append(dice)

    # Calculate the average Dice coefficient
    average_dice = np.mean(dice_scores)
    #print(f"Average Dice coefficient over the test set: {average_dice:.4f}")
    return average_dice



# Function to delete old models if they are superseded:
def delete_old_best(seed_value, path_to_saved_models):
    seed_string = str(seed_value)
    for filename in os.listdir(path_to_saved_models):
        file_path = os.path.join(path_to_saved_models, filename)
        if filename.startswith(seed_string) and filename.endswith(".tf"):
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted folder: {file_path}")
            else:
                print(f"Skipping: {file_path} is not a regular file or folder")
  
