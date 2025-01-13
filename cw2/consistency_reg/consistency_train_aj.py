import os
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
from skimage.measure import label

# Set paths and configurations
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH_TO_DATA = '/Users/antonio/YEAR_4/ML/CW2/MLMI-Group-Project/cw2/train'
SAVE_PATH = '/Users/antonio/YEAR_4/ML/CW2/MLMI-Group-Project/models'
os.makedirs(SAVE_PATH, exist_ok=True)

# Set seeds for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Data Reader Class
class DataReader:
    def __init__(self, folder):
        self.folder = folder

    def load_data(self, indices, data_type="image"):
        filenames = [f"{data_type}_train{i:03d}.npy" for i in indices]
        data = []
        for fn in filenames:
            try:
                arr = np.load(os.path.join(self.folder, fn)).astype(np.float32)
                data.append(np.expand_dims(arr, axis=-1))
            except FileNotFoundError:
                print(f"File not found: {fn}")
        data = np.stack(data)
        if data_type == "image":
            data /= 255.0
        return data

# Optimised Data Augmentation
def data_augmentation(images, labels=None, prob_flip=0.5, prob_rotate=0.5, noise_stddev=0.01):
    augmented_images = []
    augmented_labels = [] if labels is not None else None
    
    for i, img in enumerate(images):
        img = tf.convert_to_tensor(img)
        lbl = tf.convert_to_tensor(labels[i]) if labels is not None else None

        if tf.random.uniform(()) < prob_flip:
            img = tf.image.flip_left_right(img)
            if lbl is not None:
                lbl = tf.image.flip_left_right(lbl)

        if tf.random.uniform(()) < prob_rotate:
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k)
            if lbl is not None:
                lbl = tf.image.rot90(lbl, k)

        img += tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=noise_stddev, dtype=img.dtype)
        augmented_images.append(img)
        if labels is not None:
            augmented_labels.append(lbl)

    augmented_images = tf.stack(augmented_images)
    augmented_labels = tf.stack(augmented_labels) if labels is not None else None

    return augmented_images, augmented_labels

# Dice Coefficient Metric
@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Combined BCE and Dice Loss
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# UNet Model Builder
def build_unet(input_shape=(None, None, None, 1), dropout_rate=0.3):
    inputs = layers.Input(shape=input_shape)

    # Encoder Block 1
    c1 = layers.Conv3D(32, (3, 3, 3), padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.Conv3D(32, (3, 3, 3), padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.Dropout(dropout_rate)(c1)
    p1 = layers.MaxPooling3D(pool_size=(1, 2, 2))(c1)

    # Encoder Block 2
    c2 = layers.Conv3D(64, (3, 3, 3), padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)
    c2 = layers.Conv3D(64, (3, 3, 3), padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)
    c2 = layers.Dropout(dropout_rate)(c2)
    p2 = layers.MaxPooling3D(pool_size=(1, 2, 2))(c2)

    # Encoder Block 3
    c3 = layers.Conv3D(128, (3, 3, 3), padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)
    c3 = layers.Conv3D(128, (3, 3, 3), padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)
    c3 = layers.Dropout(dropout_rate)(c3)
    p3 = layers.MaxPooling3D(pool_size=(1, 2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv3D(256, (3, 3, 3), padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)
    c4 = layers.Conv3D(256, (3, 3, 3), padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)
    c4 = layers.Dropout(dropout_rate)(c4)

    # Decoder Block 1
    u1 = layers.Conv3DTranspose(128, (3, 3, 3), strides=(1, 2, 2), padding='same')(c4)
    u1 = layers.concatenate([u1, c3], axis=-1)
    c5 = layers.Conv3D(128, (3, 3, 3), padding='same')(u1)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.ReLU()(c5)

    # Decoder Block 2
    u2 = layers.Conv3DTranspose(64, (3, 3, 3), strides=(1, 2, 2), padding='same')(c5)
    u2 = layers.concatenate([u2, c2], axis=-1)
    c6 = layers.Conv3D(64, (3, 3, 3), padding='same')(u2)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.ReLU()(c6)

    # Decoder Block 3
    u3 = layers.Conv3DTranspose(32, (3, 3, 3), strides=(1, 2, 2), padding='same')(c6)
    u3 = layers.concatenate([u3, c1], axis=-1)
    c7 = layers.Conv3D(32, (3, 3, 3), padding='same')(u3)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.ReLU()(c7)

    # Output Layer
    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(c7)
    model = models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    # Load data
    data_reader = DataReader(PATH_TO_DATA)
    indices = list(range(100))
    train_indices, val_indices = indices[:80], indices[80:]
    unlabelled_indices = indices[80:]  # Simulated unlabelled data for SSL
    
    train_images = data_reader.load_data(train_indices, "image")
    train_labels = data_reader.load_data(train_indices, "label")
    val_images = data_reader.load_data(val_indices, "image")
    val_labels = data_reader.load_data(val_indices, "label")
    unlabelled_images = data_reader.load_data(unlabelled_indices, "image")

    # Define hyperparameters
    BATCH_SIZE = 4
    EPOCHS = 10
    DROPOUT_RATE = 0.3
    LAMBDA_CONSISTENCY = 1.0

    # Build U-Net model
    model = build_unet(input_shape=train_images.shape[1:], dropout_rate=DROPOUT_RATE)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=combined_loss, metrics=[dice_coefficient])

    # Train U-Net (baseline, no SSL)
    print("Training U-Net (Baseline, no SSL)...")
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=EPOCHS, batch_size=BATCH_SIZE)
    baseline_dice = model.evaluate(val_images, val_labels, verbose=0, return_dict=True)['dice_coefficient']
    print(f"Baseline U-Net Dice Coefficient: {baseline_dice:.4f}")

    # Train U-Net with SSL
    print("Training U-Net with SSL...")
    for epoch in range(EPOCHS):
        # Shuffle labelled and unlabelled data
        train_indices = np.arange(len(train_images))
        np.random.shuffle(train_indices)
        train_images = train_images[train_indices]
        train_labels = train_labels[train_indices]
        unlabelled_indices = np.arange(len(unlabelled_images))
        np.random.shuffle(unlabelled_indices)
        unlabelled_images = unlabelled_images[unlabelled_indices]

        # Training loop
        for batch_idx in range(len(train_images) // BATCH_SIZE):
            batch_images = train_images[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
            batch_labels = train_labels[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
            batch_unlabelled = unlabelled_images[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]

            # Augment data
            augmented_images, augmented_labels = data_augmentation(batch_images, batch_labels)
            unlabelled_augmented_images, _ = data_augmentation(batch_unlabelled)

            with tf.GradientTape() as tape:
                # Model predictions
                y_pred = model(batch_images, training=True)
                unlabelled_y_pred = model(batch_unlabelled, training=True)
                unlabelled_augmented_y_pred = model(unlabelled_augmented_images, training=True)

                # Loss calculation
                supervised_loss = tf.keras.losses.BinaryCrossentropy()(batch_labels, y_pred)
                consistency_loss = tf.reduce_mean(tf.square(unlabelled_y_pred - unlabelled_augmented_y_pred))
                total_loss = supervised_loss + LAMBDA_CONSISTENCY * consistency_loss

            # Apply gradients
            gradients = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Evaluate U-Net with SSL
    ssl_dice = model.evaluate(val_images, val_labels, verbose=0, return_dict=True)['dice_coefficient']
    print(f"U-Net with SSL Dice Coefficient: {ssl_dice:.4f}")

    # Compare results
    print("Comparison of U-Net Performance:")
    print(f"Baseline U-Net Dice: {baseline_dice:.4f}")
    print(f"U-Net with SSL Dice: {ssl_dice:.4f}")
