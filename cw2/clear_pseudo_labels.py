# This script removes pseudo labels (only the labels, not the images) in the 'unlabel' folder

import os

# Path to the folder containing unlabelled data
path_to_unlabel_folder = './MLMI-Group-Project/cw2/unlabel'

# Loop through all files in the directory
for filename in os.listdir(path_to_unlabel_folder):
    # Check if the file matches the pattern "label_unlabelxxx.npy"
    if filename.startswith("label_unlabel") and filename.endswith(".npy"):
        # Full path to the file
        file_path = os.path.join(path_to_unlabel_folder, filename)
        # Remove the file
        os.remove(file_path)
        print(f"Deleted: {file_path}")
        
