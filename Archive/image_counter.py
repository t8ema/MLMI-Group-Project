import os
import nibabel as nib

data_dir = "MLMI-Group-Project/cw2/data"  # Directory containing images and masks
minimum_shape = (64, 64, 32)  # Minimum required shape

# Initialize counters
valid_images_count = 0
total_images_count = 0

# Loop through the files in the directory
for file_name in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file_name)
    
    # Check if the file is a NIfTI file
    if file_name.endswith('.nii') and file_name.__contains__('img'):
        total_images_count += 1
        try:
            # Load the NIfTI image
            nifti_img = nib.load(file_path)
            img_shape = nifti_img.shape
            
            # Check if the image satisfies the minimum shape
            if all(s >= ms for s, ms in zip(img_shape, minimum_shape)):
                valid_images_count += 1
        except Exception as e:
            print(f"Error loading {file_name}: {e}")



print(f"Total NIfTI images: {total_images_count}")
print(f"Images meeting the minimum shape {minimum_shape}: {valid_images_count}")
print(f"Invalid images: {total_images_count - valid_images_count}")

# (32, 32, 16): 588 valid, 1 invalid
# (64, 64, 16): 588 valid, 1 invalid
# (64, 64, 32): 454 valid, 135 invalid
