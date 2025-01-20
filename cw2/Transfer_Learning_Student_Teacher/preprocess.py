import os
import random
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

"""
Processes data in line with expected format for train.py
Also resamples data into a given target shape
Creates a new folder called processed_data and puts images and labels there
"""
# Parameters
data_dir = "./data"  # Directory containing images and masks
train_dir = "./train_data_2d/"  # Directory for processed outputs
test_dir = "./test_data_2d/"  # Directory for processed outputs
target_shape = (64, 64, 32)  # Shape to re-process images to (x, y, number of slices) - originally used (128, 128, 32), but is faster with (64, 64, 32)
slices_first = True # Put the slice dimension first, so if you have e.g. target_shape 128,128,32, it will become 32,128,128
roi_value = 6  # Value in the mask to extract as ROI
image_normalisation = True  # Whether to normalize images between 0 and 1

SEED = 42
random.seed(SEED)

# Ensure output directory exists
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Process files
file_count = 0
train_count = 0
files = os.listdir(data_dir)  # Get the list of files
random.shuffle(files)         # Shuffle the list in place
images = [file for file in files if "_img" in file]
num_images = len(images)
train_test_split = int(0.9 * num_images)

for i, file_name in enumerate(images):
    # Skip files without .nii extension
    if not file_name.endswith(".nii"):
        print('Incorrect file type, skipping...')
        continue

    # Identify image and corresponding mask files
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

    if i < train_test_split:
        img_output_path = os.path.join(train_dir, f"image{(file_count):03d}.npy")
        mask_output_path = os.path.join(train_dir, f"label{(file_count):03d}.npy")
        train_count += 1
    else:
        img_output_path = os.path.join(test_dir, f"image{(file_count-train_count):03d}.npy")
        mask_output_path = os.path.join(test_dir, f"label{(file_count-train_count):03d}.npy")

    np.save(img_output_path, resampled_img)
    np.save(mask_output_path, binary_mask)
    print(f"(step {i}) Processed and saved: {file_name} as {os.path.basename(img_output_path)} and {os.path.basename(mask_output_path)}")
    
    file_count += 1

print(f'{file_count} files were saved.')
print('Preprocessing complete.')
