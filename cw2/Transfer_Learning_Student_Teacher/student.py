import os
import math
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers, backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import GaussianNoise

# Paths and Configurations
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH_TO_DATA = './train_data_2d/'
TEACHER_MODEL_PATH = './supervised_train_130125/modeL_epoch_71.keras'
SAVE_PATH = './models/student_11/'
os.makedirs(SAVE_PATH, exist_ok=True)

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

    def load_data_2d(self, indices, data_type="image"):
        filenames = [f"{data_type}{i:03d}.npy" for i in indices]
        slices = []
        for fn in filenames:
            try:
                volume = np.load(os.path.join(self.folder, fn)).astype(np.float32)
                volume /= 255.0 if data_type == "image" else 1.0
                for slice_idx in range(volume.shape[0]):
                    slice_2d = volume[slice_idx, :, :]
                    slices.append(np.expand_dims(slice_2d, axis=-1))
            except FileNotFoundError:
                print(f"File not found: {fn}")
        return np.stack(slices)
    
def create_dataset(images, labels, shuffle=False, augment=False, batch_size=32):
    def preprocess(image, label):
        if augment:
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                if label is not None:
                    label = tf.image.flip_left_right(label)

            angle_rad = tf.random.uniform(shape=[], minval=-np.pi / 12, maxval=np.pi / 12)  
            image = rotate_image(image, angle_rad)
            if label is not None:
                label = rotate_image(label, angle_rad)

            image = add_noise(image)

            if label is not None:
                label = tf.cast(label, tf.float32)
                label = tf.clip_by_value(label, 0.0, 1.0) 
                label = tf.cast(label > 0.5, tf.int32) 
        
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)

    return dataset

def rotate_image(image, angle_rad, interpolation='BILINEAR'):
    cos_angle = tf.math.cos(angle_rad)
    sin_angle = tf.math.sin(angle_rad)

    transform_matrix = tf.stack([
        cos_angle, -sin_angle, 0.0,
        sin_angle,  cos_angle, 0.0,
        0.0,       0.0
    ])

    transform_matrix = tf.reshape(transform_matrix, [1, 8])

    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    rotated_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),  
        transforms=transform_matrix,
        interpolation=interpolation,
        output_shape=[height, width],
        fill_value=0
    )

    return tf.squeeze(rotated_image, axis=0) 

noise_layer = GaussianNoise(0.0001)

def add_noise(image):
    image = tf.image.random_brightness(image, max_delta=0.02)
    image = tf.image.random_contrast(image, lower=0.98, upper=1.02)

    image = tf.expand_dims(image, axis=0)
    noisy_image = noise_layer(image, training=True)  # training=True ensures noise is added
    noisy_image = tf.squeeze(noisy_image, axis=0)

    return noisy_image

@tf.keras.utils.register_keras_serializable(package='Custom')
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))  
    y_pred_f = K.flatten(K.cast(y_pred, 'float32')) 
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable(package='Custom')
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def kl_divergence_loss(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 1e-7, 1.0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

    log_term = tf.math.log(y_true / y_pred)
    kl = y_true * log_term
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl, axis=-1))

    # Ensure the KL loss is non-negative
    return tf.maximum(kl_loss, 0.0)

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def bce_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

def total_loss(y_true, y_pred, dice_weight=0.5, bce_weight=0.5):
    dice = dice_loss(y_true, y_pred)
    bce = bce_loss(y_true, y_pred)
    return dice_weight * dice + bce_weight * bce

def evaluate_model(model, val_dataset, threshold=0.5):
    dice_scores = []
    
    for val_images, val_labels in val_dataset:
        predictions = model(val_images, training=False)
        pred_binary = tf.cast(predictions > threshold, tf.float32)
        dice_score = dice_coef(val_labels, pred_binary)
        dice_scores.append(dice_score)
    
    # Compute the mean Dice score across batches
    mean_dice = tf.reduce_mean(dice_scores).numpy()
    return mean_dice

def build_unet_2d(input_shape=(64, 64, 1), dropout_rate=0.3, l2_reg=1e-4):
    kernel_reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape)
    
    # Encoder: Block 1
    c1 = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=kernel_reg)(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=kernel_reg)(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.Dropout(dropout_rate)(c1)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)
    
    # Encoder: Block 2
    c2 = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=kernel_reg)(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)
    c2 = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=kernel_reg)(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)
    c2 = layers.Dropout(dropout_rate)(c2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)
    
    # Encoder: Block 3
    c3 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=kernel_reg)(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)
    c3 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=kernel_reg)(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)
    c3 = layers.Dropout(dropout_rate)(c3)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(c3)
    
    # Bottleneck (Bridge)
    c4 = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=kernel_reg)(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)
    c4 = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=kernel_reg)(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)
    c4 = layers.Dropout(dropout_rate)(c4)
    
    # Decoder: Block 1
    u1 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=kernel_reg)(c4)
    u1 = layers.concatenate([u1, c3])
    c5 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=kernel_reg)(u1)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.ReLU()(c5)
    c5 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=kernel_reg)(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.ReLU()(c5)
    c5 = layers.Dropout(dropout_rate)(c5)
    
    # Decoder: Block 2
    u2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=kernel_reg)(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=kernel_reg)(u2)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.ReLU()(c6)
    c6 = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=kernel_reg)(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.ReLU()(c6)
    c6 = layers.Dropout(dropout_rate)(c6)
    
    # Decoder: Block 3
    u3 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=kernel_reg)(c6)
    u3 = layers.concatenate([u3, c1])
    c7 = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=kernel_reg)(u3)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.ReLU()(c7)
    c7 = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=kernel_reg)(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.ReLU()(c7)
    c7 = layers.Dropout(dropout_rate)(c7)
    
    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    # Model
    model = models.Model(inputs, outputs)
    return model

def generate_pseudolabels(teacher_model, unlabeled_images, threshold=0.5):
    soft_labels = teacher_model.predict(unlabeled_images)
    hard_labels = tf.cast(soft_labels > threshold, tf.float32)
    return hard_labels

def train_student_model(student_model, train_dataset, val_dataset, epochs, patience, min_delta=1e-3, callbacks=None):
    # Initialize variables for early stopping
    best_weights = None
    best_val_loss = np.inf
    patience_counter = 0

    # Metrics storage
    dice_scores = []
    metrics_log = []

    # Initialize callbacks if provided
    if callbacks is None:
        callbacks = []

    # Set up callback states
    for callback in callbacks:
        callback.set_model(student_model)
        callback.on_train_begin()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Trigger callbacks at the start of the epoch
        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        # Training loop
        for batch_idx, (batch_images, batch_labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = student_model(batch_images, training=True)
                bce = bce_loss(batch_labels, predictions)
                dice = dice_loss(batch_labels, predictions)
                loss = total_loss(batch_labels, predictions, dice_weight=0.7, bce_weight=0.3)

            gradients = tape.gradient(loss, student_model.trainable_variables)
            student_model.optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))

            print(f'| Batch {batch_idx + 1} | BCE Loss: {bce:.4f} | Dice Loss: {dice:.4f} | Total Loss: {loss:.4f}')

        # Validation evaluation
        val_result = student_model.evaluate(val_dataset, verbose=0, return_dict=True)
        val_loss = val_result['loss']
        val_dice = val_result.get('dice_coef', np.nan)
        print(f"Validation Loss: {val_loss:.4f}, Dice Score: {val_dice:.4f}")

        # Capture the current learning rate
        current_lr = float(tf.keras.backend.get_value(student_model.optimizer.learning_rate))
        print(f"Learning Rate: {current_lr:.6f}")

        # Record metrics including learning rate
        dice_scores.append(val_dice)
        metrics_log.append({
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'learning_rate': current_lr  # Log learning rate
        })

        # Trigger callbacks after validation
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs={"val_loss": val_loss, "val_dice": val_dice})

        # Save best model weights if improvement
        if best_val_loss - val_loss > min_delta:
            best_weights = student_model.get_weights()
            best_val_loss = val_loss
            patience_counter = 0
            student_model.save(f"model_epoch_{epoch + 1}.keras")
            print(f"Model improved and saved at epoch {epoch + 1}")
        
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")

        # Early stopping check
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Trigger callbacks at the end of training
    for callback in callbacks:
        callback.on_train_end()

    # Restore best model weights
    if best_weights is not None:
        student_model.set_weights(best_weights)
        print("Best model weights restored.")

    # Save metrics to CSV
    with open('student_training_metrics.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics_log[0].keys())
        writer.writeheader()
        writer.writerows(metrics_log)
    print("Training metrics saved to 'student_training_metrics.csv'")

    # Plot Dice Score and Learning Rate over Epochs
    plt.figure(figsize=(10, 5))
    epochs_range = range(1, len(dice_scores) + 1)
    
    plt.plot(epochs_range, dice_scores, label="Validation Dice Score", marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Validation Dice Score Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("student_dice_scores.png")
    plt.close()
    print("Dice score plot saved to 'student_dice_scores.png'")

    # Learning rate plot
    learning_rates = [log['learning_rate'] for log in metrics_log]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, learning_rates, label="Learning Rate", marker='x', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("student_learning_rate.png")
    plt.close()
    print("Learning rate plot saved to 'student_learning_rate.png'")

    return student_model

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 200
    PATIENCE = 10
    INPUT_SHAPE = (64, 64, 1)
    INITIAL_LEARNING_RATE = 1e-3
    DROPOUT_RATE = 0.3

    # Load and preprocess 2D slices
    data_reader = DataReader(PATH_TO_DATA)
    all_indices = list(range(len(os.listdir(PATH_TO_DATA)) // 2))
    random.shuffle(all_indices)

    val_split = int(0.1 * len(os.listdir(PATH_TO_DATA)) // 2)
    unlabelled_split = int(0.5 * len(os.listdir(PATH_TO_DATA)) // 2)
    val_indices = all_indices[:val_split]
    train_indices = all_indices[val_split:unlabelled_split]
    unlabelled_indices = all_indices[unlabelled_split:]
    
    train_images = data_reader.load_data_2d(train_indices, "image")
    train_labels = data_reader.load_data_2d(train_indices, "label")
    val_images = data_reader.load_data_2d(val_indices, "image")
    val_labels = data_reader.load_data_2d(val_indices, "label")
    unlabelled_images = data_reader.load_data_2d(unlabelled_indices, "image")
    unlabelled_labels = data_reader.load_data_2d(unlabelled_indices, "label")

    print(f'Train images shape: {train_images.shape}')
    print(f'Train labels shape: {train_labels.shape}')
    print(f'Validation images shape: {val_images.shape}')
    print(f'Validation labels shape: {val_labels.shape}')
    print(f'Unlabelled images shape: {unlabelled_images.shape}')

    # Load the teacher model
    teacher_model = tf.keras.models.load_model(TEACHER_MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    pseudo_labels = generate_pseudolabels(teacher_model, unlabelled_images)
    
    assert pseudo_labels.shape == unlabelled_images.shape, "Shape mismatch between images and pseudolabels"

    pseudo_labels_dice = dice_coef(unlabelled_labels,pseudo_labels)
    print(f'Pseudolabels Dice Score: {pseudo_labels_dice}')

    # Create datasets using tf.data pipeline
    train_dataset = create_dataset(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True, augment=False)
    val_dataset = create_dataset(val_images, val_labels, batch_size=BATCH_SIZE, shuffle=False, augment=False)
    unlabelled_dataset = create_dataset(unlabelled_images, pseudo_labels, batch_size=BATCH_SIZE, shuffle=True, augment=True)

    # Define optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True
    )

    tf.keras.backend.clear_session()  # Clear previous sessions
    student_model = build_unet_2d(input_shape=INPUT_SHAPE, dropout_rate=DROPOUT_RATE)
    optimizer = tf.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    student_model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy', dice_coef])
    
    # Define ReduceLROnPlateau callback
    lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Can also monitor 'val_dice' or 'accuracy'
        factor=0.1,  # Reduces LR by a factor of 0.5
        patience=2,  # Number of epochs with no improvement after which LR will be reduced
        min_delta=1e-3,
        cooldown=1,
        verbose=1,  # Print a message when LR is reduced
        min_lr=1e-7  # Minimum learning rate
    )

    # Train the student model
    train_student_model(student_model, 
                        unlabelled_dataset,
                        val_dataset,
                        epochs=EPOCHS,
                        patience=PATIENCE,
                        callbacks=[lr_reduction])  # Include ReduceLROnPlateau in callbacks

    # Evaluate and save
    test_dice = evaluate_model(student_model, val_dataset)
    print(f"Validation Dice Score: {test_dice:.4f}")
    student_model.save(os.path.join(SAVE_PATH, "student_2d_model.keras"))
