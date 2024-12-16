# MLMI-Group-Project
#### General info:
Make sure the ***unzipped*** data files are placed under the 'data' folder after you clone the repository to your own environment
Run data.py (or new_data.py, if it now correctly orders the slices), then train.py, then view_results.py
<br />
train.py automatically saves some examples of the test data results to the results folder. These can be viewed using view_results.py
<br />
<br />
  
## data.py:
Processes the data into the expected format for train.py. Resamples into a consistent shape
<br />
<br />

## model.py:
Trains a model on the image and mask data to create a segmentation prediction model
<br />
<br />

## view_results.py
See the results of the trained model in 3D (scroll through slices)
<br />
<br />

## visualise.py
See the results of the trained model - better to use view_results.py over this, because this one heavily downsamples the image before allowing you to see it, so subtle details are lost. This script will likely be archived soon.
<br />
<br />

## view_niftii_3d.py:
Plots some of the raw data for observation
<br />
<br />

## view_processed_image.py:
Plots some of the processed data for observation
