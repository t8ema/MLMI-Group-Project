# This script pre-processes the data for a specified number of files, e.g. 50
# All images are normalised, and the mask for the rectum is selected and isolated
# The value of the mask is set to 1, and the background is set to 0 - binary mask
# The number of slices can also be specified (not all images have the same number of slices)
# Images that don't have at least the specified number of slices are skipped


import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom



# Parameters
data_dir = "MLMI-Group-Project/cw2/data"  # Directory containing images and masks
output_dir = "MLMI-Group-Project/cw2/processed_data"  # Directory for processed outputs
num_files_to_process = 50  # Number of image-mask pairs to process
target_shape = (128, 128, 24)  # Final shape (x, y, number of slices)
roi_value = 6  # Value in the mask to extract as ROI
# 6 is the rectum, and the one selected for our purposes



# Create output directories
image_output_dir = os.path.join(output_dir, "image")
mask_output_dir = os.path.join(output_dir, "mask")
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)



# Helper function to normalize image data to range [0, 1]
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))



# Process files
processed_count = 0
for file_name in sorted(os.listdir(data_dir)):
    # Skip files without .nii extension
    if not file_name.endswith(".nii"):
        continue

    # Identify image and corresponding mask files
    if "_img" in file_name:
        base_name = file_name.replace("_img.nii", "")
        mask_file_name = f"{base_name}_mask.nii"
        img_path = os.path.join(data_dir, file_name)
        mask_path = os.path.join(data_dir, mask_file_name)

        # Check if corresponding mask file exists
        if not os.path.exists(mask_path):
            print(f"Mask file for {file_name} not found, skipping...")
            continue

        # Load image and mask
        img = nib.load(img_path)
        mask = nib.load(mask_path)

        img_data = img.get_fdata()
        mask_data = mask.get_fdata()

        # Skip files if the number of slices is insufficient
        if img_data.shape[2] < target_shape[2]:
            print(f"{file_name} has fewer slices ({img_data.shape[2]}) than required ({target_shape[2]}), skipping...")
            continue

        # Resample image and mask to target shape
        scale_factors = (
            target_shape[0] / img_data.shape[0],
            target_shape[1] / img_data.shape[1],
            1,  # Slices scaling will be handled by slicing
        )
        resampled_img = zoom(img_data, scale_factors, order=3)  # Cubic interpolation
        resampled_mask = zoom(mask_data, scale_factors, order=0)  # Nearest-neighbor interpolation

        # Select the middle slices
        middle_slice_idx = img_data.shape[2] // 2
        slice_start = middle_slice_idx - target_shape[2] // 2
        slice_end = middle_slice_idx + target_shape[2] // 2
        resampled_img = resampled_img[:, :, slice_start:slice_end]
        resampled_mask = resampled_mask[:, :, slice_start:slice_end]

        # Skip if slicing went out of bounds
        if resampled_img.shape[2] != target_shape[2]:
            print(f"{file_name} could not be resized correctly to {target_shape}, skipping...")
            continue

        # Normalize image
        normalized_img = normalize_image(resampled_img)

        # Create a binary mask for the specified ROI value
        binary_mask = (resampled_mask == roi_value).astype(np.uint8)

        # Save as NumPy files
        img_output_path = os.path.join(image_output_dir, f"image_{processed_count}.npy")
        mask_output_path = os.path.join(mask_output_dir, f"mask_{processed_count}.npy")

        np.save(img_output_path, normalized_img)
        np.save(mask_output_path, binary_mask)

        print(f"Processed and saved: {file_name} and {mask_file_name} as image_{processed_count}.npy and mask_{processed_count}.npy")

        processed_count += 1

        # Stop processing if the required number of files is reached
        if processed_count >= num_files_to_process:
            break
