# Makes prediction by ensemble (multiple models)
# Averages model predictions to create labels

import os
import tensorflow as tf
import numpy as np

# Paths to the models
model_paths = [
    './MLMI-Group-Project/cw2/saved_models/USED_unet_step_110.tf',
    './MLMI-Group-Project/cw2/saved_models/USED_unet_step_158.tf'
]

# Path to folder with unlabelled data
path_to_unlabel_folder = './MLMI-Group-Project/cw2/unlabel'
# Path to save predicted labels
output_folder = './MLMI-Group-Project/cw2/unlabel'

size_minibatch = 6  # Size of minibatch on which the model was trained



# Load all models
loaded_models = [tf.saved_model.load(path) for path in model_paths]

print('-------------------------------------------------')

# Remove pseudo labels (only the labels, not the images) in the 'unlabel' folder:
for filename in os.listdir(path_to_unlabel_folder):
    if filename.startswith("label_unlabel") and filename.endswith(".npy"):
        file_path = os.path.join(path_to_unlabel_folder, filename)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

if __name__ == "__main__":
    # Get a sorted list of all .npy files in the folder
    unlabelled_images = sorted([f for f in os.listdir(path_to_unlabel_folder) if f.endswith('.npy')])

    for i, image_name in enumerate(unlabelled_images):  # Loop through all files
        # Generate file paths
        image_file = os.path.join(path_to_unlabel_folder, image_name)
        output_file = os.path.join(output_folder, f"label_{image_name.replace('image_', '')}")

        # Load and preprocess the image
        image = np.load(image_file)
        gray_chan_image = np.expand_dims(image, axis=-1)
        batch_dim_image = np.expand_dims(gray_chan_image, axis=0)
        stacked_image = np.tile(batch_dim_image, (size_minibatch, 1, 1, 1, 1))  # Duplicate the image to fit batch size

        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(stacked_image, dtype=tf.float32)

        # Collect predictions from all models
        predictions = []
        for model in loaded_models:
            pred = model(input_tensor)[0, :, :, :, 0].numpy()
            predictions.append(pred)

        # Average predictions
        averaged_prediction = np.mean(predictions, axis=0)

        # Binarize the prediction (thresholding at 0.5)
        final_prediction = (averaged_prediction > 0.5).astype(np.uint8)

        # Save the prediction as a .npy file
        np.save(output_file, final_prediction)
        print(f"Saved averaged pseudo-label to {output_file}")
      
