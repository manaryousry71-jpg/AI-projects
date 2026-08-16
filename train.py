"""
Music Generation AI
Model Training Module

This script:
1. Loads preprocessed data.
2. Builds an LSTM model.
3. Trains the model with validation.
4. Uses EarlyStopping to prevent overfitting.
5. Saves the best trained model.
"""

# Import Required Libraries
import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)

# Configuration
DATA_FOLDER = "data"
MODEL_FOLDER = "model"
MODEL_NAME = "music_lstm.keras"
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)

# Load Data
def load_data():
    network_input = np.load(
        os.path.join(
            DATA_FOLDER,
            "network_input.npy"
        )
    )
    network_output = np.load(
        os.path.join(
            DATA_FOLDER,
            "network_output.npy"
        )
    )
    return (
        network_input,
        network_output
    )
# Build LSTM Model
def build_model(
    input_shape,
    output_size
):
    model = Sequential()
    # First LSTM layer
    model.add(
        LSTM(
            256,
            input_shape=input_shape,
            return_sequences=True
        )
    )
    model.add(
        Dropout(0.3)
    )
    # Second LSTM layer
    model.add(
        LSTM(
            256
        )
    )
    model.add(
        Dropout(0.3)
    )
    # Fully connected layer
    model.add(
        Dense(
            256,
            activation="relu"
        )
    )
    model.add(
        Dropout(0.3)
    )
    # Output layer
    model.add(
        Dense(
            output_size,
            activation="softmax"
        )
    )
    return model


# Train Model
def train_model(
    model,
    network_input,
    network_output
):
    # Compile model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    # Save the best model
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            MODEL_FOLDER,
            MODEL_NAME
        ),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    # Stop training if validation loss
    # stops improving
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    # Start training
    history = model.fit(
        network_input,
        network_output,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[
            checkpoint,
            early_stopping
        ],
        shuffle=True
    )
    return history


# Main
def main():
    print("=" * 60)
    print("Loading Dataset...")
    network_input, network_output = load_data()
    print(
        f"Input Shape: {network_input.shape}"
    )
    print(
        f"Output Shape: {network_output.shape}"
    )
    print("=" * 60)
    print("Building LSTM Model...")
    output_size = (
        int(np.max(network_output)) + 1
    )
    model = build_model(
        network_input.shape[1:],
        output_size
    )
    model.summary()
    print("=" * 60)
    print("Training Started...")
    train_model(
        model,
        network_input,
        network_output
    )
    print("=" * 60)
    print("Training Finished Successfully!")
    print(
        f"Best model saved to: "
        f"{MODEL_FOLDER}/{MODEL_NAME}"
    )
    print("=" * 60)


# Run Program
if __name__ == "__main__":
    main()