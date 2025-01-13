# **IMPORTANT INFO:**
- Make sure the UNZIPPED data files are placed under the 'data' folder after you clone the repository to your own environment <br />
- Make sure you rename any models you want to keep in saved_models before you run train.py, otherwise these models will be overwritten <br />
<br />
<br />

# **Running order:**
### SSL-pseudo: <br />

data.py -> train_supervised.py  -> ensemble_validator.py/ensemble_label_and_train.py -> single_model_validator.py 
<br />
<br />
(See information below for more info on each script) 
<br />
<br />
  
# **data.py:**
Processes the data into the expected format for train.py. Resamples into a consistent shape, and normalises images between 0 and 1. Puts data into train, val, test and unlabel folders. Images in the unlabel folder are stored without labels, so are the unlabelled data that we can make predictions on and use in the training data for later models
<br />
<br />

# **train_supervised.py:**
Trains the model (fully supervised, i.e. on the data that already has labels) and saves trained models in the saved_models folder. These models can be loaded with the model_loader script (to view results for a single image). Tests the model on the specified test dataset, and stops training early if there is no improvement in the dice (patience parameter)
<br />
<br />

# **ensemble_validator.py:**
Allows ensemble models (combined averaged predictions from several models) to be validated by calculating an average dice value over the validation set
<br />
<br />

# **ensemble_label_and_train.py:**
Allows you to pseudo-label data using specified saved models (or just one model, if you choose to), and trains on specified combinations of labelled and pseudo labelled images
<br />
<br />

# **single_model_validator.py:**
Validates one model at a time
<br />
<br />

# **Visualisation_scripts:**
### **view_niftii_3d.py:**
Plots some of the raw data for observation
<br />

### **view_processed_image.py:**
Plots some of the processed data for observation
<br />
