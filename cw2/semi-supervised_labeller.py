# This script generates and saves labels for unlabelled data using a given saved model



import os

import tensorflow as tf
import numpy as np



model_save_path = './MLMI-Group-Project/cw2/saved_models/residual_unet_step_50.tf' # Path to the model you want to load
path_to_unlabel_folder = './MLMI-Group-Project/cw2/unlabel' # Path to folder with unlabelled data
output_folder = './MLMI-Group-Project/cw2/unlabel'  # Path to save predicted labels



# Choose the loaded model
loaded_model = tf.saved_model.load(model_save_path)



print('-------------------------------------------------')



# Remove pseudo labels (only the labels, not the images) in the 'unlabel' folder:
# Loop through all files in the directory
for filename in os.listdir(path_to_unlabel_folder):
    # Check if the file matches the pattern "label_unlabelxxx.npy"
    if filename.startswith("label_unlabel") and filename.endswith(".npy"):
        # Full path to the file
        file_path = os.path.join(path_to_unlabel_folder, filename)
        # Remove the file
        os.remove(file_path)
        print(f"Deleted: {file_path}")



if __name__ == "__main__":
    # Get a sorted list of all .npy files in the folder
    unlabelled_images = sorted([f for f in os.listdir(path_to_unlabel_folder) if f.endswith('.npy')])

    for i, image_name in enumerate(unlabelled_images):  # Loop through all files
        # Generate file paths
        image_file = os.path.join(path_to_unlabel_folder, image_name)
        output_file = os.path.join(output_folder, f"label_{image_name}")

        # Load and preprocess the image
        image = np.load(image_file)
        gray_chan_image = np.expand_dims(image, axis=-1)
        batch_dim_image = np.expand_dims(gray_chan_image, axis=0)
        stacked_image = np.tile(batch_dim_image, (4, 1, 1, 1, 1))  # Duplicate the image to fit batch size of 4

        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(stacked_image, dtype=tf.float32)

        # Make predictions (and convert to numpy)
        pred = loaded_model(input_tensor)[0, :, :, :, 0].numpy()

        # Save the prediction as a .npy file
        np.save(output_file, pred)
        print(f"Saved predicted label to {output_file}")
