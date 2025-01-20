import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Set seeds for reproducibility
SEED = 40
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

class DataReader:
    def __init__(self, folder):
        self.folder = folder

    def load_data_2d(self, indices, file_number, data_type="image"):
        filename = f"{data_type}{int(file_number):03d}.npy"
        slices = [] 
        try:
            volume = np.load(os.path.join(self.folder, filename)).astype(np.float32)
            volume /= 255.0 if data_type == "image" else 1.0
            for slice_idx in range(volume.shape[0]):
                slice_2d = volume[slice_idx, :, :]
                slices.append(np.expand_dims(slice_2d, axis=-1))
        except FileNotFoundError:
            print(f"File not found: {filename}")
        return np.stack(slices)

def preprocess_image(image_path):
    img = np.load(image_path).astype(np.float32)
    img /= 255.0
    slices = [np.expand_dims(img[i], axis=-1) for i in range(img.shape[0])]
    img = np.stack(slices, axis=0)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_and_display(model, PATH_TO_DATA):
    # Load and preprocess 2D slices
    data_reader = DataReader(PATH_TO_DATA)
    all_indices = list(range(len(os.listdir(PATH_TO_DATA)) // 2))
    random.shuffle(all_indices)

    file_number = input('Enter the image number: ')

    val_split = int(0.1 * len(os.listdir(PATH_TO_DATA)) // 2)
    val_indices = all_indices[:val_split]
    val_images = data_reader.load_data_2d(val_indices, file_number, "image")
    val_labels = data_reader.load_data_2d(val_indices, file_number, "label")

    print(f'Validation images shape: {val_images.shape}')
    print(f'Validation labels shape: {val_labels.shape}')

    num_slices = val_images.shape[0]

    print(f'Number of slices: {num_slices}')

    pred_list = []

    # Predict each slice
    for i in range(num_slices):
        slice_2d = val_images[i]  # Corrected slicing for batch dimension
        pred_slice = model.predict(np.expand_dims(slice_2d, axis=0))  # Add batch dim for prediction
        pred_binary = (pred_slice > 0.5).astype(np.uint32)  # Binary thresholding
        pred_list.append(pred_binary)

    pred_label = np.stack(pred_list, axis=0)
    pred_label = np.squeeze(pred_label, axis=1)

    val_images = tf.cast(val_images, tf.float32)
    val_labels = tf.cast(val_labels, tf.float32)
    pred_label = tf.cast(pred_label, tf.float32)

    print(f'True labels shape: {val_labels.shape}')
    print(f'Predictions shape: {pred_label.shape}')
    
    # Calculate Dice score (now using val_labels and pred_label)
    average_dice = evaluate_model(model, val_images, val_labels)
    print('Average dice score over validation set:', average_dice)

    # Set up the display
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    # Display first slice (you can use a slider to change slice index)
    slice_idx = 0
    img_display = ax[0].imshow(val_images[slice_idx], cmap="gray")
    label_display = ax[1].imshow(val_labels[slice_idx], cmap="gray")
    pred_display = ax[2].imshow(pred_label[slice_idx], cmap="gray")

    ax[0].set_title("Original Image")
    ax[1].set_title("Original Label")
    ax[2].set_title("Predicted Label")
    for a in ax:
        a.axis("off")

    # Slider for navigating through slices
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        slice_idx = int(slider.val)
        img_display.set_data(val_images[slice_idx])
        label_display.set_data(val_labels[slice_idx])
        pred_display.set_data(pred_label[slice_idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    average_dice = evaluate_model(model, val_images, val_labels)

    print(f'Average dice for validation set: {average_dice}')

@tf.keras.utils.register_keras_serializable(package='Custom')
def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_pred * y_true)
    union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
    return 2 * intersection / (union + smooth)

@tf.keras.utils.register_keras_serializable(package='Custom')
def dice_loss(y_true, y_pred):
    coef = dice_coef(y_true, y_pred)
    return 1 - coef

def evaluate_model(model, val_images, val_labels, threshold=0.5):
    dice_scores = []
    
    for image, label in zip(val_images, val_labels):
        image = tf.expand_dims(image, axis=0)  
        prediction = model(image, training=False)
        pred_binary = tf.cast(prediction > threshold, tf.float32)
        pred_binary = tf.squeeze(pred_binary, axis=0)
        dice_score = dice_coef(label, pred_binary)
        dice_scores.append(dice_score)

    mean_dice = tf.reduce_mean(dice_scores).numpy()
    return mean_dice

# Main execution
if __name__ == "__main__":
    model_path = './models/student_9/model_epoch_19.keras'
    model = load_model(model_path, custom_objects={'dice_coef': dice_coef})
    PATH_TO_DATA = './train_data_2d/' 
    predict_and_display(model, PATH_TO_DATA)

