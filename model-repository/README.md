Put successful models here, i.e. models with high average dice values over the validation set. Please use the naming convention specified below. When you upload a model, please add it to the model list below the naming convention with any comments i.e. how many training images were used, what the learning rate (and learning rate decay) were, and if it is a semi-supervised model, specify how many fully labelled and how many pseudo labelled images were used.
</br>
</br>

# Naming convention:
<p> {supervised/semi-supervised}_residual_unet_{number of training steps}_{loss}_{average dice over validation set}
</br>
</br>

Hence if you trained a supervised model (a model trained only on the fully labelled training data), for 128 steps with a final step loss of 0.270449825, and an average dice over the validation set of 0.7015, please name the file: supervised_residual_unet_128_2704_7015.tf (remove the 0. at the start of the decimal values, and round to 4 decimal places)
</br>
</br>

If you trained a semi-supervised model (a model trained on a mix of fully labelled and pseudo-labelled data), for 256 steps with a final step loss of 0.346123909, average validation dice of 0.6984, please name the file: semi-supervised_residual_unet_256_3461_6984.tf
</br>
</br>

### ***Examples:***
Model type: semi-supervised </br>
Steps: 126 </br>
Final step loss: 0.271562091 </br>
Validation dice: 0.7186 </br>
Name: semi-supervised_residual_unet_126_2716_7186.tf </br>
</br>

Model type: supervised </br>
Steps: 100 </br>
Final step loss: 0.125617902 </br>
Validation dice: 0.6570 </br>
Name: semi-supervised_residual_unet_100_1256_6570.tf </br>
</br>


# Model list:

