import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, backend as K
from sklearn.metrics import roc_auc_score

# Paths
BASE_PATH = "/Users/antonio/YEAR_4/ML/CW2/MLMI-Group-Project/cw2"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH = os.path.join(BASE_PATH, "val")
TEST_PATH = os.path.join(BASE_PATH, "test")
UNLABEL_PATH = os.path.join(BASE_PATH, "unlabel")

SAVE_PATH = os.path.join(BASE_PATH, "models")
os.makedirs(SAVE_PATH, exist_ok=True)

# Data Reader Class
class DataReader:
    def __init__(self, folder):
        self.folder = folder

    def load_data(self, data_type="image"):
        filenames = sorted(
            [f for f in os.listdir(self.folder) if f.startswith(data_type) and f.endswith(".npy")]
        )
        data = [np.load(os.path.join(self.folder, fn)) for fn in filenames]
        data = np.stack(data)
        if data_type.startswith("image"):
            data /= 255.0  # Normalize images
        return np.expand_dims(data, axis=-1)  # Add channel dimension


# Initialize readers
train_reader = DataReader(TRAIN_PATH)
val_reader = DataReader(VAL_PATH)
test_reader = DataReader(TEST_PATH)
unlabel_reader = DataReader(UNLABEL_PATH)

# Load datasets
train_images = train_reader.load_data("image_train")
train_labels = train_reader.load_data("label_train")
val_images = val_reader.load_data("image_val")
val_labels = val_reader.load_data("label_val")
test_images = test_reader.load_data("image_test")
test_labels = test_reader.load_data("label_test")
unlabel_images = unlabel_reader.load_data("image_unlabel")

# Check dataset shapes
print(f"Train images: {train_images.shape}, Train labels: {train_labels.shape}")
print(f"Validation images: {val_images.shape}, Validation labels: {val_labels.shape}")
print(f"Test images: {test_images.shape}, Test labels: {test_labels.shape}")
print(f"Unlabelled images: {unlabel_images.shape}")

# Hyperparameters
BATCH_SIZE = 4
EPOCHS = 10
DROPOUT_RATE = 0.4
LAMBDA_CONSISTENCY = 1.0

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
    y_true_f = K.cast(K.flatten(y_true), dtype="float32")  # Cast to float32
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

# Initialize metrics tracking
supervised_dice_scores = []
supervised_roc_auc_scores = []
ssl_dice_scores = []
ssl_roc_auc_scores = []

# Build and compile models
supervised_model = build_unet(input_shape=train_images.shape[1:], dropout_rate=DROPOUT_RATE)
supervised_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=combined_loss, metrics=[dice_coefficient])

ssl_model = build_unet(input_shape=train_images.shape[1:], dropout_rate=DROPOUT_RATE)
ssl_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=combined_loss, metrics=[dice_coefficient])

# Supervised Training
print("Training Supervised U-Net...")
for epoch in range(EPOCHS):
    supervised_model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=1, batch_size=BATCH_SIZE, verbose=1
    )
    val_metrics = supervised_model.evaluate(val_images, val_labels, verbose=0, return_dict=True)
    supervised_dice_scores.append(val_metrics['dice_coefficient'])

    # Calculate ROC-AUC
    val_predictions = supervised_model.predict(val_images).flatten()
    val_labels_flat = val_labels.flatten()
    val_roc_auc = roc_auc_score(val_labels_flat, val_predictions)
    supervised_roc_auc_scores.append(val_roc_auc)

    print(f"Epoch {epoch + 1}/{EPOCHS}")

# SSL Training
print("Training SSL U-Net...")
for epoch in range(EPOCHS):
    for batch_idx in range(len(train_images) // BATCH_SIZE):
        batch_images = train_images[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
        batch_labels = train_labels[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
        batch_unlabelled = unlabel_images[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]

        augmented_images, augmented_labels = data_augmentation(batch_images, batch_labels)
        unlabelled_augmented_images, _ = data_augmentation(batch_unlabelled)

        with tf.GradientTape() as tape:
            y_pred = ssl_model(augmented_images, training=True)
            unlabelled_y_pred = ssl_model(batch_unlabelled, training=True)
            unlabelled_augmented_y_pred = ssl_model(unlabelled_augmented_images, training=True)

            supervised_loss = tf.keras.losses.BinaryCrossentropy()(augmented_labels, y_pred)
            consistency_loss = tf.reduce_mean(tf.square(unlabelled_y_pred - unlabelled_augmented_y_pred))
            total_loss = supervised_loss + LAMBDA_CONSISTENCY * consistency_loss

        gradients = tape.gradient(total_loss, ssl_model.trainable_variables)
        ssl_model.optimizer.apply_gradients(zip(gradients, ssl_model.trainable_variables))

    # Validation metrics for SSL
    val_metrics_ssl = ssl_model.evaluate(val_images, val_labels, verbose=0, return_dict=True)
    ssl_dice_scores.append(val_metrics_ssl['dice_coefficient'])

    # Calculate ROC-AUC for SSL
    val_predictions_ssl = ssl_model.predict(val_images).flatten()
    val_roc_auc_ssl = roc_auc_score(val_labels.flatten(), val_predictions_ssl)
    ssl_roc_auc_scores.append(val_roc_auc_ssl)

    print(f"Epoch {epoch + 1}/{EPOCHS} - SSL Validation Dice: {val_metrics_ssl['dice_coefficient']:.4f}, ROC AUC: {val_roc_auc_ssl:.4f}")


# Plot Dice and ROC-AUC metrics
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 6))

# Dice Coefficient Plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, supervised_dice_scores, label="Supervised Dice", marker="o")
plt.plot(epochs_range, ssl_dice_scores, label="SSL Dice", marker="x")
plt.title("Validation Dice Coefficient per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Dice Coefficient")
plt.legend()
plt.grid(True)

# ROC-AUC Plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, supervised_roc_auc_scores, label="Supervised ROC-AUC", marker="o")
plt.plot(epochs_range, ssl_roc_auc_scores, label="SSL ROC-AUC", marker="x")
plt.title("Validation ROC-AUC per Epoch")
plt.xlabel("Epochs")
plt.ylabel("ROC-AUC")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
