import os
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import backend as K

# Set paths and configurations
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH_TO_DATA = './train_data_2d/'
SAVE_PATH = './models/'
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


@tf.keras.utils.register_keras_serializable(package='Custom')
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable(package='Custom')
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

@tf.keras.utils.register_keras_serializable(package='Custom')
def combined_dice_bce_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return alpha * d_loss + beta * bce

def evaluate_model(model, test_data, threshold=0.5):
    val_images, val_labels = test_data
    predictions = model.predict(val_images)
    dice_scores = []
    for pred, truth in zip(predictions, val_labels):
        pred_binary = (pred > threshold).astype(np.float32)
        dice_score = dice_coef(truth, pred_binary)
        dice_scores.append(dice_score)
    return np.mean(dice_scores)

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

def train_model(model, train_data, val_data, batch_size, epochs, patience, min_delta=1e-3):

    train_images, train_labels = train_data
    val_images, val_labels = val_data

    # Initialize variables for early stopping
    best_weights = None
    best_val_loss = np.inf
    patience_counter = 0

    # Metrics storage
    dice_scores = []
    metrics_log = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Shuffle training data
        indices = np.arange(len(train_images))
        np.random.shuffle(indices)
        train_images = train_images[indices]
        train_labels = train_labels[indices]

        num_batches = len(train_images) // batch_size
        print(f'Number of batches: {num_batches}')

        for batch_idx in range(num_batches):
            # Prepare batch data
            batch_images = train_images[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_labels = train_labels[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = model(batch_images, training=True)

                # Compute supervised Dice loss
                loss = dice_loss(batch_labels, y_pred)

            # Apply gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print(f'Batch {batch_idx + 1}/{num_batches} complete. Loss: {loss.numpy():.4f}')

        # Validation evaluation
        val_result = model.evaluate(val_images, val_labels, verbose=0, return_dict=True)
        val_loss = val_result['loss']
        val_dice = val_result.get('dice_coef', np.nan)

        print(f"Validation Loss: {val_loss:.4f}, Dice Score: {val_dice:.4f}")

        # Record metrics
        dice_scores.append(val_dice)
        metrics_log.append({
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'val_dice': val_dice
        })

        # Save best model weights if significant improvement
        if best_val_loss - val_loss > min_delta:
            best_weights = model.get_weights()
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience if improvement is significant
            model.save(f"model_epoch_{epoch + 1}.keras")
            print(f"Model improved by more than {min_delta} and saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            print(f"No significant improvement (Δ < {min_delta}). Patience counter: {patience_counter}/{patience}")

        # Early stopping check
        if patience_counter >= patience:
            print("Early stopping triggered due to insufficient improvement.")
            break

    # Restore best model weights
    if best_weights is not None:
        model.set_weights(best_weights)
        print("Best model weights restored.")

    # Save metrics to CSV
    with open('supervised_training_metrics.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics_log[0].keys())
        writer.writeheader()
        writer.writerows(metrics_log)
    print("Training metrics saved to 'supervised_training_metrics.csv'")

    # Plot Dice Score over Epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(dice_scores) + 1), dice_scores, label="Validation Dice Score", marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Validation Dice Score Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("supervised_dice_scores.png")
    plt.close()
    print("Dice score plot saved to 'supervised_dice_scores.png'")

    return model

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 500
    PATIENCE = 20
    INPUT_SHAPE = (64, 64, 1)
    INITIAL_LEARNING_RATE = 5e-5
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

    print(f'Train images shape: {train_images.shape}')
    print(f'Train labels shape: {train_labels.shape}')
    print(f'Validation images shape: {val_images.shape}')
    print(f'Validation labels shape: {val_labels.shape}')
    print(f'Unlabelled images shape: {unlabelled_images.shape}')

    # Define a simple exponential learning rate decay schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,  # Initial learning rate
        decay_steps=100,                               # After 100 steps, decay
        decay_rate=0.96,                               # Reduce by 4% every decay step
        staircase=True                                 # Decay steps are integer multiples
    )

    tf.keras.backend.clear_session()  # Clear previous sessions
    model = build_unet_2d(INPUT_SHAPE, DROPOUT_RATE)
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy', dice_coef])

    # Train the 2D model
    train_model(model,
                train_data=(train_images, train_labels),
                val_data=(val_images, val_labels),
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                patience=PATIENCE)
    
    # Evaluate and save
    test_dice = evaluate_model(model, (val_images, val_labels))
    print(f"Validation Dice Score: {test_dice:.4f}")
    model.save(os.path.join(SAVE_PATH, "unet_2d_model.keras"))
