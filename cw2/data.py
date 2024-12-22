# Processes data in line with expected format for train.py
# Creates a new folder called processed_data and puts images and labels there
# Also resamples data into a given target shape
# test data are also created with labels for testing of the model later (see visualise results)

# TO DO: Add a validation set, so we can validate the model on completely unseen data

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom



# Parameters
data_dir = "MLMI-Group-Project/cw2/data"  # Directory containing images and masks
output_dir = "MLMI-Group-Project/cw2/processed_data"  # Directory for processed outputs
num_train = 50  # Number of training images
num_test = 30  # Number of test images
target_shape = (64, 64, 32)  # Shape to re-process images to (x, y, number of slices) - originally used (128, 128, 32), but is faster with (64, 64, 32)
slices_first = True # Put the slice dimension first, so if you have e.g. target_shape 128,128,32, it will become 32,128,128
roi_value = 6  # Value in the mask to extract as ROI
image_normalisation = True  # Whether to normalize images between 0 and 1



# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Check if output directory is empty
if os.path.exists(output_dir) and os.listdir(output_dir):
    raise ValueError(f"Output directory '{output_dir}' is not empty! \n Please delete all files in '{output_dir}' before running this script.")



# Process files
train_count = 0
test_count = 0
total_processed = 0

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

        # Normalize image if the parameter is enabled
        if image_normalisation:
            resampled_img = (resampled_img - np.min(resampled_img)) / (np.max(resampled_img) - np.min(resampled_img))

        # Create a binary mask for the specified ROI value
        binary_mask = (resampled_mask == roi_value).astype(np.uint8)

        # Rearrange axes to make slices the first dimension
        if slices_first == True:
            resampled_img = np.transpose(resampled_img, (2, 0, 1))  # From (x, y, slices) to (slices, x, y)
            binary_mask = np.transpose(binary_mask, (2, 0, 1))      # Match the same reordering

        # Determine whether to save as train or test
        if total_processed < num_train:
            img_output_path = os.path.join(output_dir, f"image_train{train_count:02d}.npy")
            mask_output_path = os.path.join(output_dir, f"label_train{train_count:02d}.npy")
            train_count += 1

            np.save(img_output_path, resampled_img)
            np.save(mask_output_path, binary_mask)
            print(f"Processed and saved: {file_name} as {os.path.basename(img_output_path)} and {os.path.basename(mask_output_path)}")
        elif total_processed < num_train + num_test:
            img_output_path = os.path.join(output_dir, f"image_test{test_count:02d}.npy")
            mask_output_path = os.path.join(output_dir, f"label_test{test_count:02d}.npy")#########################################
            test_count += 1

            np.save(img_output_path, resampled_img)
            np.save(mask_output_path, binary_mask)
            print(f"Processed and saved: {file_name} as {os.path.basename(img_output_path)} and {os.path.basename(mask_output_path)}")
            #print(f"Processed and saved: {file_name} as {os.path.basename(img_output_path)}")
        else:
            # Stop processing once test images are completed
            break

        total_processed += 1
