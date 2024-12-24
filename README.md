# **IMPORTANT INFO:**
<br />
- Make sure the ***unzipped*** data files are placed under the 'data' folder after you clone the repository to your own environment
- Run data.py, then train.py, then model_validator.py
- Make sure you rename any models you want to keep in saved_models before you run train.py, otherwise these models will be overwritten
- Make sure you empty the train, val and unlabel folders before re-running the data.py script
<br />
<br />
  
# **data.py:**
Processes the data into the expected format for train.py. Resamples into a consistent shape, and normalises images between 0 and 1. Puts data into train, val and unlabel folders. Images in the unlabel folder are stored withoput labels, so are the unlabelled data that we can make predictions on and use in the training data for later models.
<br />
<br />

# **train.py:**
Trains the model and saves trained models in the saved_models folder. These models can be loaded with the model_loader script (to view results for a single image)
<br />
<br />

# **model_validator.py:**
Allows models to be validated by calculating an average dice value over the validation set
<br />
<br />

# **Visualisation_scripts:**
### **view_niftii_3d.py:**
Plots some of the raw data for observation
<br />

### **view_processed_image.py:**
Plots some of the processed data for observation
<br />
