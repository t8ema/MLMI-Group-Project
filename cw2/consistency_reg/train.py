import os
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks, regularizers

# Set paths and configurations
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH_TO_DATA = './train_data/'
SAVE_PATH = './models/'
os.makedirs(SAVE_PATH, exist_ok=True)

# Set seeds for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

def build_unet(input_shape=(None, None, None, 1), dropout_rate=0.3):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder: Block 1
    c1 = layers.Conv3D(32, (3, 3, 3), padding='same')(inputs)  # Changed padding to 'same'
    c1 = layers.ReLU()(c1)
    c1 = layers.Conv3D(32, (3, 3, 3), padding='same')(c1)  # Changed padding to 'same'
    c1 = layers.ReLU()(c1)
    c1 = layers.Dropout(dropout_rate)(c1)  # Add dropout
    p1 = layers.MaxPooling3D(pool_size=(1, 2, 2))(c1)
    
    # Encoder: Block 2
    c2 = layers.Conv3D(64, (3, 3, 3), padding='same')(p1)  # Changed padding to 'same'
    c2 = layers.ReLU()(c2)
    c2 = layers.Conv3D(64, (3, 3, 3), padding='same')(c2)  # Changed padding to 'same'
    c2 = layers.ReLU()(c2)
    c2 = layers.Dropout(dropout_rate)(c2)  # Add dropout
    p2 = layers.MaxPooling3D(pool_size=(1, 2, 2))(c2)
    
    # Encoder: Block 3
    c3 = layers.Conv3D(128, (3, 3, 3), padding='same')(p2)  # Changed padding to 'same'
    c3 = layers.ReLU()(c3)
    c3 = layers.Conv3D(128, (3, 3, 3), padding='same')(c3)  # Changed padding to 'same'
    c3 = layers.ReLU()(c3)
    c3 = layers.Dropout(dropout_rate)(c3)  # Add dropout
    p3 = layers.MaxPooling3D(pool_size=(1, 2, 2))(c3)
    
    # Bottleneck (Bridge) Block
    c4 = layers.Conv3D(256, (3, 3, 3), padding='same')(p3)  # Changed padding to 'same'
    c4 = layers.ReLU()(c4)
    c4 = layers.Conv3D(256, (3, 3, 3), padding='same')(c4)  # Changed padding to 'same'
    c4 = layers.ReLU()(c4)
    c4 = layers.Dropout(dropout_rate)(c4)  # Add dropout
    
    # Decoder: Block 1
    u1 = layers.Conv3DTranspose(128, (3, 3, 3), strides=(1, 2, 2), padding='same')(c4)
    u1 = layers.Concatenate()([u1, c3])
    c5 = layers.Conv3D(128, (3, 3, 3), padding='same')(u1)
    c5 = layers.ReLU()(c5)
    c5 = layers.Conv3D(128, (3, 3, 3), padding='same')(c5)
    c5 = layers.ReLU()(c5)
    c5 = layers.Dropout(dropout_rate)(c5)  # Add dropout
    
    # Decoder: Block 2
    u2 = layers.Conv3DTranspose(64, (3, 3, 3), strides=(1, 2, 2), padding='same')(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = layers.Conv3D(64, (3, 3, 3), padding='same')(u2)
    c6 = layers.ReLU()(c6)
    c6 = layers.Conv3D(64, (3, 3, 3), padding='same')(c6)
    c6 = layers.ReLU()(c6)
    c6 = layers.Dropout(dropout_rate)(c6)  # Add dropout
    
    # Decoder: Block 3
    u3 = layers.Conv3DTranspose(32, (3, 3, 3), strides=(1, 2, 2), padding='same')(c6)
    u3 = layers.Concatenate()([u3, c1])
    c7 = layers.Conv3D(32, (3, 3, 3), padding='same')(u3)
    c7 = layers.ReLU()(c7)
    c7 = layers.Conv3D(32, (3, 3, 3), padding='same')(c7)
    c7 = layers.ReLU()(c7)
    c7 = layers.Dropout(dropout_rate)(c7)  # Add dropout
    
    # Output layer
    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(c7)
    
    # Build and return the model
    model = models.Model(inputs, outputs)
    return model
    
class DataReader:
    def __init__(self, folder):
        self.folder = folder

    def load_data(self, indices, data_type="image"):
        filenames = [f"{data_type}{i:03d}.npy" for i in indices]
        data = []
        for fn in filenames:
            try:
                arr = np.load(os.path.join(self.folder, fn)).astype(np.float32)  # Convert to float32
                data.append(np.expand_dims(arr, axis=-1))
            except FileNotFoundError:
                print(f"File not found: {fn}")
        data = np.stack(data)
        # Normalize to [0, 1] range if it's image data
        if data_type == "image":
            data /= 255.0
        return data

# Test function
def evaluate_model(model, test_data, threshold=0.5):
    predictions = model.predict(test_data[0])
    dice_scores = []
    for pred, truth in zip(predictions, test_data[1]):
        pred_binary = (pred > threshold).astype(np.uint8)
        intersection = np.sum(pred_binary * truth)
        union = np.sum(pred_binary) + np.sum(truth)
        dice_score = 2 * intersection / union if union > 0 else 1.0
        dice_scores.append(dice_score)
    return np.mean(dice_scores)

# Data augmentation function
def data_augmentation(images, labels=None, prob_flip=0.5, prob_rotate=0.5, noise_stddev=0.01):    
    augmented_images = []
    augmented_labels = [] if labels is not None else None
    
    for i, img in enumerate(images):
        img = tf.convert_to_tensor(img)
        lbl = tf.convert_to_tensor(labels[i]) if labels is not None else None

        # Random flipping (left-right)
        if tf.random.uniform(()) < prob_flip:
            img = tf.image.flip_left_right(img)
            if lbl is not None:
                lbl = tf.image.flip_left_right(lbl)
        
        # Random rotation (90-degree increments)
        if tf.random.uniform(()) < prob_rotate:
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)  # Random rotations: 0, 90, 180, 270
            img = tf.image.rot90(img, k)
            if lbl is not None:
                lbl = tf.image.rot90(lbl, k)
        
        # Add Gaussian noise to the image
        img += tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=noise_stddev, dtype=img.dtype)

        augmented_images.append(img)
        if labels is not None:
            augmented_labels.append(lbl)
    
    augmented_images = tf.stack(augmented_images)
    augmented_labels = tf.stack(augmented_labels) if labels is not None else None
    
    return augmented_images, augmented_labels

# Combined loss function (Supervised + Consistency Loss)
def combined_loss(y_true, y_pred, augmented_y_pred, lambda_consistency=1.0):
    # Supervised loss (binary cross-entropy)
    supervised_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    
    # Consistency loss
    consistency_loss_val = tf.reduce_mean(tf.square(y_pred - augmented_y_pred))
    
    # Combined loss
    total_loss = supervised_loss + lambda_consistency * consistency_loss_val
    
    return total_loss

def train_model(model, train_data, unlabelled_images, val_data, batch_size, epochs, patience, lambda_consistency=1.0):
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    best_weights = None  # Variable to store the best model weights

    for epoch in range(epochs):
        # Shuffle training data
        train_indices = np.arange(len(train_images))
        np.random.shuffle(train_indices)
        train_images = train_images[train_indices]
        train_labels = train_labels[train_indices]

        # Shuffle unlabelled data separately
        unlabelled_indices = np.arange(len(unlabelled_images))
        np.random.shuffle(unlabelled_indices)
        unlabelled_images = unlabelled_images[unlabelled_indices]

        num_batches = len(train_images) // batch_size
        print(f'Number of batches: {num_batches}')

        for batch_idx in range(num_batches):
            # Labeled batch
            batch_images = train_images[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_labels = train_labels[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            # Unlabelled batch
            unlabelled_start = batch_idx * batch_size % len(unlabelled_images)
            unlabelled_end = unlabelled_start + batch_size
            batch_unlabelled = unlabelled_images[unlabelled_start:unlabelled_end]

            # Data augmentation
            augmented_images, augmented_labels = data_augmentation(batch_images, batch_labels)
            unlabelled_augmented_images, _ = data_augmentation(batch_unlabelled, None)
            
            with tf.GradientTape() as tape:
                # Model predictions
                y_pred = model(batch_images, training=True)
                augmented_y_pred = model(augmented_images, training=True)
                unlabelled_y_pred = model(batch_unlabelled, training=True)
                unlabelled_augmented_y_pred = model(unlabelled_augmented_images, training=True)

                # Compute the supervised loss
                supervised_loss = model.compiled_loss(tf.convert_to_tensor(batch_labels), tf.convert_to_tensor(y_pred))
                consistency_loss = tf.reduce_mean(tf.square(unlabelled_y_pred - unlabelled_augmented_y_pred))
                total_loss = supervised_loss + lambda_consistency * consistency_loss
            
            # Gradient update
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print(f'Batch {batch_idx} complete. Supervised loss: {supervised_loss}, Consistency loss: {consistency_loss}, Total loss: {total_loss}')
        
        # Validation evaluation
        val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
        print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Store the best weights
        if best_weights is None or val_loss < best_val_loss:
            best_weights = model.get_weights()
            best_val_loss = val_loss
        
        # # Early stopping logic
        # early_stopping.on_epoch_end(epoch, {'val_loss': val_loss})
        # if early_stopping.stopped_epoch > 0:
        #     print("Early stopping triggered.")
        #     break

    # Restore best weights if early stopping was triggered
    if best_weights is not None:
        model.set_weights(best_weights)

    return model

if __name__ == "__main__":

    # Hyperparameters
    BATCH_SIZE = 6
    EPOCHS = 10
    PATIENCE = 10
    INPUT_SHAPE = (6, 48, 48, 1)
    LEARNING_RATE = 5e-5
    LAMBDA_CONSISTENCY = 1.0

    # Load and split data
    data_reader = DataReader(PATH_TO_DATA)
    all_indices = list(range(len(os.listdir(PATH_TO_DATA)) // 2))
    random.shuffle(all_indices)

    val_split = int(0.1 * len(all_indices))
    unlabelled_split = int(0.5 * len(all_indices))
    val_indices = all_indices[:val_split]
    train_indices = all_indices[val_split:unlabelled_split]
    unlabelled_indices = all_indices[unlabelled_split:]

    train_images = data_reader.load_data(train_indices, "image")
    train_labels = data_reader.load_data(train_indices, "label")
    val_images = data_reader.load_data(val_indices, "image")
    val_labels = data_reader.load_data(val_indices, "label")
    unlabelled_images = data_reader.load_data(unlabelled_indices, "image")

    # Build and compile the model
    tf.keras.backend.clear_session()
    unet_model = build_unet(INPUT_SHAPE)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    unet_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    print(f'Train images shape: {train_images.shape}')
    print(f'Train labels shape: {train_labels.shape}')
    print(f'Unlabelled images shape: {unlabelled_images.shape}')
    print(unet_model.summary())

    # Train model
    trained_model = train_model(
        model=unet_model,
        train_data=(train_images, train_labels),
        unlabelled_images=unlabelled_images,
        val_data=(val_images, val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        lambda_consistency=LAMBDA_CONSISTENCY
    )

    # Evaluate and save model
    test_dice = evaluate_model(trained_model, (val_images, val_labels))
    print(f"Validation Dice Score: {test_dice:.4f}")
    trained_model.save(os.path.join(SAVE_PATH, "unet_model_with_consistency"))
    print("Model with consistency regularization saved.")
