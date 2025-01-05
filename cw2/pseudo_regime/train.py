# Train for semi-supervised experiment
# This version has a patience parameter to stop training early if there is no improvement in the dice over the test set



import os
import random

import tensorflow as tf
import numpy as np



os.environ["CUDA_VISIBLE_DEVICES"]="0"
path_to_data = './MLMI-Group-Project/cw2/train'
path_to_test = './MLMI-Group-Project/cw2/test'
SAVE_PATH = './MLMI-Group-Project/cw2/saved_models'
os.makedirs(SAVE_PATH, exist_ok=True)  # Create SAVE_PATH directory if it doesn't exist



# Set seeds for reproducibility
SEED = 42  # Original is 40
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Enable deterministic operations in TensorFlow
os.environ["TF_DETERMINISTIC_OPS"] = "1"



# Define dice coefficient calculator
def calculate_dice_coefficient(predicted, truth):
    """
    Calculate the Dice coefficient for 3D binary masks.

    Parameters:
        predicted (numpy.ndarray): Predicted binary mask (3D).
        truth (numpy.ndarray): Ground truth binary mask (3D).

    Returns:
        float: Dice coefficient.
    """
    intersection = np.sum(predicted * truth)
    volume_sum = np.sum(predicted) + np.sum(truth)
    if volume_sum == 0:
        return 1.0  # Perfect score if both masks are empty
    return (2.0 * intersection) / volume_sum

# Define data tester
def test_model(model_path, path_to_test_folder, binarize_mask=True, binary_threshold=0.5, minibatch_size=8):
    """
    Test the model on unseen data and calculate the average Dice coefficient.

    Parameters:
        model_path (str): Path to the saved model.
        path_to_test_folder (str): Path to the folder containing test images and labels.
        binarize_mask (bool): Whether to binarize the predicted masks.
        binary_threshold (float): Threshold for binarization.
        batch_size (int): Number of images in a batch.

    Returns:
        float: Average Dice coefficient over the test set.
    """
    # Load the model
    model = tf.saved_model.load(model_path)

    # Get the list of all image files in the test folder
    image_files = sorted([f for f in os.listdir(path_to_test_folder) if f.startswith("image_test") and f.endswith(".npy")])
    num_images = len(image_files)

    dice_scores = []

    # Process test data in batches
    for start_idx in range(0, num_images, minibatch_size):
        end_idx = min(start_idx + minibatch_size, num_images)
        current_batch_size = end_idx - start_idx

        # Prepare batch data
        batch_images = []
        batch_truths = []

        for i in range(start_idx, end_idx):
            # Load the image
            image_file = os.path.join(path_to_test_folder, f"image_test{i:03d}.npy")
            truth_mask_file = os.path.join(path_to_test_folder, f"label_test{i:03d}.npy")

            image = np.load(image_file)
            truth_mask = np.load(truth_mask_file)

            gray_chan_image = np.expand_dims(image, axis=-1)
            batch_images.append(gray_chan_image)
            batch_truths.append(truth_mask)

        # Stack images into a single batch
        batch_images = np.stack(batch_images, axis=0)
        batch_truths = np.stack(batch_truths, axis=0)

        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(batch_images, dtype=tf.float32)

        # If the number of images is not divisible by the minibatch size, the final test minibatch will be 
        # unequal to the minibatch size. Hence, we make sure here that the first dimension of the tensor is
        # the correct size (minibatch size). If not, we go back to the start of the loop
        if input_tensor.shape[0] != minibatch_size:
            break
        
        # Make predictions
        predictions = model(input_tensor)

        for j in range(current_batch_size):
            pred = predictions[j, :, :, :, 0].numpy()

            if binarize_mask:
                pred = (pred > binary_threshold).astype(np.uint8)

            # Calculate the Dice coefficient
            dice = calculate_dice_coefficient(pred, batch_truths[j])
            dice_scores.append(dice)

    # Calculate the average Dice coefficient
    average_dice = np.mean(dice_scores)
    #print(f"Average Dice coefficient over the test set: {average_dice:.4f}")
    return average_dice



## Define functions for network layers
def conv3d(input, filters, downsample=False, activation=True, batch_norm=False):
    if downsample: strides = [1,2,2,2,1]
    else: strides = [1,1,1,1,1]
    y = tf.nn.conv3d(input, filters, strides=strides, padding='SAME')
    if batch_norm: y = batch_norm(y)
    if activation: y = tf.nn.relu(y)
    return y  # where bn can be added

def resnet_block(input, filters, batch_norm=False):
    y = conv3d(input, filters[..., 0])
    y = conv3d(y, filters[..., 1], activation=False) + input
    if batch_norm: y = batch_norm(y)
    return tf.nn.relu(y)  # where bn can be added

def downsample_maxpool(input, filters):
    y = conv3d(input, filters)
    return tf.nn.max_pool3d(y, ksize=[1,3,3,3,1], padding='SAME', strides=[1,2,2,2,1])

def deconv3d(input, filters, out_shape, batch_norm=False):
    y = tf.nn.conv3d_transpose(input, filters, output_shape=out_shape, strides=[1,2,2,2,1], padding='SAME')
    if batch_norm: y = batch_norm(y)
    return tf.nn.relu(y)  # where bn can be added

def batch_norm(inputs, is_training, decay = 0.999):
    # This is where to insert the implementation of batch normalisaiton
    return inputs

def add_variable(var_shape, var_list, var_name=None, initialiser=None):
    if initialiser is None:
        initialiser = tf.initializers.glorot_normal()
    if var_name is None:
        var_name = 'var{}'.format(len(var_list))
        var_list.append(tf.Variable(initialiser(var_shape), name=var_name, trainable=True))
    return var_list

## Define a model (a 3D U-Net variant) with residual layers with trainable weights
# ref: https://arxiv.org/abs/1512.03385  & https://arxiv.org/abs/1505.04597
num_channels = 32
nc = [num_channels*(2**i) for i in range(4)]
var_list=[]
# intial-layer
var_list = add_variable([5,5,5,1,nc[0]], var_list)
# encoder-s0
var_list = add_variable([3,3,3,nc[0],nc[0],2], var_list)
var_list = add_variable([3,3,3,nc[0],nc[0],2], var_list)
var_list = add_variable([3,3,3,nc[0],nc[0]], var_list)
var_list = add_variable([3,3,3,nc[0],nc[1]], var_list)
# encoder-s1
var_list = add_variable([3,3,3,nc[1],nc[1],2], var_list)
var_list = add_variable([3,3,3,nc[1],nc[1],2], var_list)
var_list = add_variable([3,3,3,nc[1],nc[1]], var_list)
var_list = add_variable([3,3,3,nc[1],nc[2]], var_list)
# encoder-s2
var_list = add_variable([3,3,3,nc[2],nc[2],2], var_list)
var_list = add_variable([3,3,3,nc[2],nc[2],2], var_list)
var_list = add_variable([3,3,3,nc[2],nc[2]], var_list)
var_list = add_variable([3,3,3,nc[2],nc[3]], var_list)
# deep-layers-s3
var_list = add_variable([3,3,3,nc[3],nc[3],2], var_list)
var_list = add_variable([3,3,3,nc[3],nc[3],2], var_list)
var_list = add_variable([3,3,3,nc[3],nc[3],2], var_list)
# decoder-s2
var_list = add_variable([3,3,3,nc[2],nc[3]], var_list)
var_list = add_variable([3,3,3,nc[2],nc[2],2], var_list)
var_list = add_variable([3,3,3,nc[2],nc[2],2], var_list)
# decoder-s1
var_list = add_variable([3,3,3,nc[1],nc[2]], var_list)
var_list = add_variable([3,3,3,nc[1],nc[1],2], var_list)
var_list = add_variable([3,3,3,nc[1],nc[1],2], var_list)
# decoder-s0
var_list = add_variable([3,3,3,nc[0],nc[1]], var_list)
var_list = add_variable([3,3,3,nc[0],nc[0],2], var_list)
var_list = add_variable([3,3,3,nc[0],nc[0],2], var_list)
# output-layer
var_list = add_variable([3,3,3,nc[0],1], var_list)

## model with corresponding layers
@tf.function
def residual_unet(input):
    # initial-layer
    skip_layers = []
    layer = conv3d(input, var_list[0])
    # encoder-s0
    layer = resnet_block(layer, var_list[1])
    layer = resnet_block(layer, var_list[2])
    skip_layers.append(layer)
    layer = downsample_maxpool(layer, var_list[3])
    layer = conv3d(layer, var_list[4])
    # encoder-s1
    layer = resnet_block(layer, var_list[5])
    layer = resnet_block(layer, var_list[6])
    skip_layers.append(layer)
    layer = downsample_maxpool(layer, var_list[7])
    layer = conv3d(layer, var_list[8])
    # encoder-s2
    layer = resnet_block(layer, var_list[9])
    layer = resnet_block(layer, var_list[10])
    skip_layers.append(layer)
    layer = downsample_maxpool(layer, var_list[11])
    layer = conv3d(layer, var_list[12])
    # deep-layers-s3
    layer = resnet_block(layer, var_list[13])
    layer = resnet_block(layer, var_list[14])
    layer = resnet_block(layer, var_list[15])
    # decoder-s2
    layer = deconv3d(layer, var_list[16], skip_layers[2].shape) + skip_layers[2]
    layer = resnet_block(layer, var_list[17])
    layer = resnet_block(layer, var_list[18])
    # decoder-s1
    layer = deconv3d(layer, var_list[19], skip_layers[1].shape) + skip_layers[1]
    layer = resnet_block(layer, var_list[20])
    layer = resnet_block(layer, var_list[21])
    # decoder-s0
    layer = deconv3d(layer, var_list[22], skip_layers[0].shape) + skip_layers[0]
    layer = resnet_block(layer, var_list[23])
    layer = resnet_block(layer, var_list[24])
    # output-layer
    layer = tf.sigmoid(conv3d(layer, var_list[25], activation=False))
    return layer

## loss function
def loss_dice(pred, target, eps=1e-6):
    dice_numerator = 2 * tf.reduce_sum(pred*target, axis=[1,2,3,4])
    dice_denominator = eps + tf.reduce_sum(pred, axis=[1,2,3,4]) + tf.reduce_sum(target, axis=[1,2,3,4])
    return  1 - tf.reduce_mean(dice_numerator/dice_denominator)


## npy data loader class
class DataReader:
    def __init__(self, folder_name):
        self.folder_name = folder_name
    def load_images_train(self, indices_mb):
        return self.load_npy_files(["image_train%03d.npy" % idx for idx in indices_mb])
    def load_images_test(self, indices_mb):
        return self.load_npy_files(["image_test%03d.npy" % idx for idx in indices_mb])
    def load_labels_train(self, indices_mb):
        return self.load_npy_files(["label_train%03d.npy" % idx for idx in indices_mb])
    def load_npy_files(self, file_names):
        images = [np.float32(np.load(os.path.join(self.folder_name, fn))) for fn in file_names]
        #return np.expand_dims(np.stack(images, axis=0), axis=4)[:, ::2, ::2, ::2, :]
        return np.expand_dims(np.stack(images, axis=0), axis=4)[:, :, :, :, :]


## training
@tf.function
def train_step(model, weights, optimizer, x, y):
    with tf.GradientTape() as tape:
        # g_tape.watched(var_list): trainable variables are automatically "watched".
        loss = loss_dice(model(x), y)
    gradients = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(gradients, weights))
    return loss

# Calculate number of training images
image_files = sorted([f for f in os.listdir(path_to_data) if f.startswith("image_train") and f.endswith(".npy")])
num_images = len(image_files)
#num_images = 8 #####################################################################################################
print('Number of training images: ', num_images)

# Model parameters
learning_rate = 7e-5  # Initial learning rate
total_iter = 1000  # Total number of iterations
freq_print = 1  # How often to print
freq_test = 5 # How often to test (and save) the model
n = num_images  # Number of training image-label pairs
patience = 10  # Number of test intervals to wait for improvement
size_minibatch = 6

num_minibatch = int(n/size_minibatch)  # Number of minibatches in each epoch
indices_train = [i for i in range(n)]
DataFeeder = DataReader(path_to_data)

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=10,
    decay_rate=0.5,
    staircase=True
)
#optimizer = tf.optimizers.Adam(learning_rate_schedule)
optimizer = tf.optimizers.Adam(learning_rate)  # If not using scheduler



class ResidualUNet(tf.Module):
    def __init__(self, var_list):
        super(ResidualUNet, self).__init__()
        self.var_list = var_list  # Store the variable list as a class attribute

    @tf.function
    def __call__(self, input):
        return residual_unet(input)  # Call the residual_unet function



# Invoke the model    
residual_unet_model = ResidualUNet(var_list)



# Set up highest dice parameter
highest_dice = 0.0  # Keep track of the highest Dice score
highest_dice_step = 0  # Step number where the highest Dice was achieved
patience_counter = 0  # Counter for patience

# Start training
for step in range(total_iter):

    # Shuffle tye data for each new set of minibatches
    if step in range(0, total_iter, num_minibatch):
        random.shuffle(indices_train)

    # find out data indices for a minibatch
    minibatch_idx = step % num_minibatch  # minibatch index
    indices_mb = indices_train[minibatch_idx*size_minibatch:(minibatch_idx+1)*size_minibatch]
    
    input_mb = DataFeeder.load_images_train(indices_mb)
    label_mb = DataFeeder.load_labels_train(indices_mb)
    # update the variables
    loss_train = train_step(residual_unet_model, var_list, optimizer, input_mb, label_mb)

    # print training information
    step1 = step+1  # for display and count
    if (step1 % freq_print) == 0:
        tf.print('Step', step1, 'loss:', loss_train)
    
    # Test and save the model periodically
    if step1 % freq_test == 0:
        model_save_path = os.path.join(SAVE_PATH, f"residual_unet_step_{step1}.tf")
        tf.saved_model.save(residual_unet_model, model_save_path)
        print(f"Model saved to {model_save_path}")
        test_dice = test_model(model_save_path, path_to_test, minibatch_size= size_minibatch)
        print(f"Average dice over test data: {test_dice:.4f}")
                
        # Check if Dice score improved
        if test_dice > highest_dice:
            highest_dice = test_dice
            highest_dice_step = step1
            patience_counter = 0  # Reset patience counter
            print(f"New highest Dice score: {highest_dice:.4f} at step {highest_dice_step}")
        else:
            patience_counter += 1  # Increment patience counter
            print(f"No improvement in Dice score. Patience counter: {patience_counter}/{patience}")
        
        # Check if patience is exceeded
        if patience_counter >= patience:
            print(f"Early stopping triggered at step {step1}. Highest Dice: {highest_dice:.4f} at step {highest_dice_step}")
            break

# Save final model
model_save_path = os.path.join(SAVE_PATH, f"residual_unet_step_finalstep.tf")
tf.saved_model.save(residual_unet_model, model_save_path)
print(f"Model saved to {model_save_path}")

print('Training done.')
