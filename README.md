# MLMI-Group-Project
#### General info:
Make sure the ***unzipped*** data files are placed under the 'data' folder after you clone the repository to your own environment
Run data.py, then train.py, then view_results.py
<br />
<br />
- data.py preprocessed the data, e.g. makes the image shapes consistent, normalisation etc
- train.py trains the model and saves the results of the test_data (i.e. the predicted mask for a random sample of test data). These test data results are stored in the results folder, under the format 'label_test{image number}_step{step number}-tf.npy', e.g. 'label_test21_step000128-tf.npy', so if you run the model for 128 steps, you could find a file called 'label_test{image number}_step000128-tf.npy', where {image number} is the number of a random image selected from the data, such as 21 or 29, etc
- view_results.py allows you to see these results by choosing the names of the file to view
<br />
<br />

***IMPORTANT:***
<br />
<br />
- Make sure you empty the results folder (delete all files inside it) before you run train.py, otherwise you will still have the results of the previous model you ran
<br />
- Make sure you empty the processed image folder before running data.py
  
## data.py:
Processes the data into the expected format for train.py. Resamples into a consistent shape, and normalises images between 0 and 1
<br />
<br />

## model.py:
Trains a model on the image and mask data to create a segmentation prediction model. Also saves predictions on test data into the results folder
<br />
<br />

## view_results.py
See the results of the trained model in 3D (scroll through slices)
<br />
<br />

## view_niftii_3d.py:
Plots some of the raw data for observation
<br />
<br />

## view_processed_image.py:
Plots some of the processed data for observation
<br />
<br />
