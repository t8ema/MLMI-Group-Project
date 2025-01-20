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
SAVE_PATH = './models/mean_teacher_1/'
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
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(images)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    def preprocess(image, label=None, augment=False):
        image = tf.cast(image, tf.float32) / 255.0  # Normalize images
        if label is not None:
            label = tf.cast(label, tf.float32)

        if augment:
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                if label is not None:
                    label = tf.image.flip_left_right(label)

            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_up_down(image)
                if label is not None:
                    label = tf.image.flip_up_down(label)

            angle_rad = tf.random.uniform(shape=[], minval=-np.pi / 12, maxval=np.pi / 12)
            image = rotate_image(image, angle_rad)
            if label is not None:
                label = rotate_image(label, angle_rad)

            image = add_noise(image)
            pass

        if label is not None:
            # Preprocess label
            label = tf.cast(label, tf.float32)
            label = tf.clip_by_value(label, 0.0, 1.0)
            label = tf.cast(label > 0.5, tf.int32)

        return (image, label) if label is not None else image
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
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

noise_layer = GaussianNoise(0.0003)

def add_noise(image):
    image = tf.image.random_brightness(image, max_delta=0.02)
    image = tf.image.random_contrast(image, lower=0.98, upper=1.02)

    image = tf.expand_dims(image, axis=0)
    noisy_image = noise_layer(image, training=True)  # training=True ensures noise is added
    noisy_image = tf.squeeze(noisy_image, axis=0)

    return noisy_image

@tf.keras.utils.register_keras_serializable(package='Custom')
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.cast(K.flatten(y_true), dtype='float32')
    y_pred_f = K.cast(K.flatten(y_pred), dtype='float32')
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

def dice_bce_loss(y_true, y_pred, dice_weight=0.6, bce_weight=0.4):
    dice = dice_loss(y_true, y_pred)
    bce = bce_loss(y_true, y_pred)
    return dice_weight * dice + bce_weight * bce

def total_loss(y_true, y_pred, teacher_pred, lambda_consistency=1.0):
    supervised_loss = dice_bce_loss(y_true, y_pred) if y_true is not None else 0
    consistency_loss = mse_loss(teacher_pred, y_pred)
    loss = supervised_loss + lambda_consistency * consistency_loss
    return loss, supervised_loss, consistency_loss

def evaluate_model(model, val_dataset, threshold=0.5):
    dice_scores = []
    for val_images, val_labels in val_dataset:
        predictions = model(val_images, training=False)
        pred_binary = tf.cast(predictions > threshold, tf.float32)
        dice_score = dice_coef(val_labels, pred_binary)
        dice_scores.append(dice_score)
    mean_dice = tf.reduce_mean(dice_scores).numpy()
    return mean_dice

def generate_pseudolabels(teacher_model, unlabeled_images, threshold=0.5):
    soft_labels = teacher_model.predict(unlabeled_images)
    hard_labels = tf.cast(soft_labels > threshold, tf.float32)
    return hard_labels

def build_unet_2d(input_shape=(64, 64, 1), dropout_rate=0.3):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder: Block 1
    c1 = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.Conv2D(32, (3, 3), padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.Dropout(dropout_rate)(c1)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)
    
    # Encoder: Block 2
    c2 = layers.Conv2D(64, (3, 3), padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)
    c2 = layers.Conv2D(64, (3, 3), padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)
    c2 = layers.Dropout(dropout_rate)(c2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)
    
    # Encoder: Block 3
    c3 = layers.Conv2D(128, (3, 3), padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)
    c3 = layers.Conv2D(128, (3, 3), padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)
    c3 = layers.Dropout(dropout_rate)(c3)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(c3)
    
    # Bottleneck (Bridge)
    c4 = layers.Conv2D(256, (3, 3), padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)
    c4 = layers.Conv2D(256, (3, 3), padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)
    c4 = layers.Dropout(dropout_rate)(c4)
    
    # Decoder: Block 1
    u1 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c4)
    u1 = layers.concatenate([u1, c3])
    c5 = layers.Conv2D(128, (3, 3), padding='same')(u1)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.ReLU()(c5)
    c5 = layers.Conv2D(128, (3, 3), padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.ReLU()(c5)
    c5 = layers.Dropout(dropout_rate)(c5)
    
    # Decoder: Block 2
    u2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(64, (3, 3), padding='same')(u2)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.ReLU()(c6)
    c6 = layers.Conv2D(64, (3, 3), padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.ReLU()(c6)
    c6 = layers.Dropout(dropout_rate)(c6)
    
    # Decoder: Block 3
    u3 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c6)
    u3 = layers.concatenate([u3, c1])
    c7 = layers.Conv2D(32, (3, 3), padding='same')(u3)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.ReLU()(c7)
    c7 = layers.Conv2D(32, (3, 3), padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.ReLU()(c7)
    c7 = layers.Dropout(dropout_rate)(c7)
    
    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    # Model
    model = models.Model(inputs, outputs)
    return model

def train_mean_teacher(student_model, teacher_model, labeled_dataset, unlabeled_dataset, val_dataset, optimizer, save_path, num_epochs=50, lambda_consistency=1.0, beta=0.99, patience=5):
    # Initialize metrics log and early stopping variables
    metrics_log = []
    best_mean_dice = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_labeled_loss = 0.0
        epoch_unlabeled_loss = 0.0
        num_labeled_batches = 0
        num_unlabeled_batches = 0

        # Train on labeled data
        for (labeled_images_batch, labeled_labels_batch) in labeled_dataset:
            with tf.GradientTape() as tape:
                student_preds = student_model(labeled_images_batch, training=True)
                loss, supervised_loss, consistency_loss = total_loss(labeled_labels_batch, student_preds, student_preds, lambda_consistency=lambda_consistency)  # For labeled data
            grads = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

            epoch_labeled_loss += loss.numpy()
            num_labeled_batches += 1

            # Print loss for the current batch
            print(f"Labeled Batch {num_labeled_batches}, Supervised Loss: {supervised_loss:.4f}, Consistency Loss: {consistency_loss:.4f}, Total Loss: {loss:.4f}")

        # Train on unlabeled data
        for unlabeled_images_batch in unlabeled_dataset:
            with tf.GradientTape() as tape:
                student_preds = student_model(unlabeled_images_batch, training=True)
                teacher_preds = teacher_model(unlabeled_images_batch, training=False)
                loss, supervised_loss, consistency_loss = total_loss(None, student_preds, teacher_preds, lambda_consistency)  # No true labels for unlabeled data
            grads = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

            epoch_unlabeled_loss += loss.numpy()
            num_unlabeled_batches += 1

            # Print loss for the current batch
            print(f"Unlabeled Batch {num_unlabeled_batches}, Supervised Loss: {supervised_loss:.4f}, Consistency Loss: {consistency_loss:.4f}, Total Loss: {loss:.4f}")

        # Average losses
        avg_labeled_loss = epoch_labeled_loss / num_labeled_batches if num_labeled_batches > 0 else 0.0
        avg_unlabeled_loss = epoch_unlabeled_loss / num_unlabeled_batches if num_unlabeled_batches > 0 else 0.0

        # Print average losses for the epoch
        print(f"Epoch {epoch+1} Labeled Loss: {avg_labeled_loss}, Unlabeled Loss: {avg_unlabeled_loss}")

        # Update teacher model (moving average of student weights)
        student_weights = student_model.get_weights()
        teacher_weights = teacher_model.get_weights()
        new_teacher_weights = [beta * teacher_weights[i] + (1 - beta) * student_weights[i] for i in range(len(student_weights))]
        teacher_model.set_weights(new_teacher_weights)

        # Evaluate the model on validation set
        mean_dice = evaluate_model(student_model, val_dataset)
        print(f"Mean Dice Score: {mean_dice}")

        # Log metrics
        metrics_log.append({
            "epoch": epoch + 1,
            "avg_labeled_loss": avg_labeled_loss,
            "avg_unlabeled_loss": avg_unlabeled_loss,
            "mean_dice": mean_dice,
        })

        # Early stopping logic
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            epochs_without_improvement = 0
            print("Improved validation Dice score. Resetting patience.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs.")
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    # Save metrics log to CSV
    results_file = os.path.join(save_path, 'training_metrics.csv')
    with open(results_file, mode="w", newline="") as csv_file:
        fieldnames = ["epoch", "avg_labeled_loss", "avg_unlabeled_loss", "mean_dice"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_log)
    
    print(f"Metrics log saved to {save_path}")
    return student_model, teacher_model  # Returning the models after training

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 200
    PATIENCE = 40
    INPUT_SHAPE = (64, 64, 1)
    INITIAL_LEARNING_RATE = 1e-4
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

    # Create datasets using tf.data pipeline
    labelled_dataset = create_dataset(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_dataset = create_dataset(val_images, val_labels, batch_size=BATCH_SIZE, shuffle=False, augment=False)
    unlabelled_dataset = create_dataset(unlabelled_images, None, batch_size=BATCH_SIZE, shuffle=True, augment=True)

    # Build models (student and teacher are initially the same)
    student_model = build_unet_2d(input_shape=(64, 64, 1), dropout_rate=DROPOUT_RATE)
    teacher_model = build_unet_2d(input_shape=(64, 64, 1), dropout_rate=DROPOUT_RATE)

    # Copy initial weights from student to teacher
    teacher_model.set_weights(student_model.get_weights())

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    
    # EMA decay rate
    beta = 0.99  # Decay rate for teacher model's moving average

    # Call the training function
    student_model, teacher_model = train_mean_teacher(
        student_model,
        teacher_model,
        labelled_dataset,
        unlabelled_dataset,
        val_dataset,
        optimizer,
        save_path=SAVE_PATH,
        num_epochs=EPOCHS, 
        patience=PATIENCE,
        lambda_consistency=1.0,
        beta=0.99
    )

    # Save models
    student_model.save(os.path.join(SAVE_PATH, "student_model.keras"))
    teacher_model.save(os.path.join(SAVE_PATH, "teacher_model.keras"))
