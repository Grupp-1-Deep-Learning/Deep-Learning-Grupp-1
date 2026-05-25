# TWO CNN MODELS
# 1. DIGIT MODEL  -> MNIST
# 2. LETTER MODEL -> EMNIST LETTERS

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"{file_path} is empty")
    if df.shape[1] < 785:
        raise ValueError(f"{file_path} does not have 785 columns")
    return df

def build_cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    return model


def train_cnn_model(X, y, num_classes, model_name, save_path):
    X = X / 255.0
    X = X.reshape(-1, 28, 28, 1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = build_cnn_model(num_classes)

    print(f"\nTraining {model_name}...")

    model.fit(X_train,y_train,epochs=10,batch_size=128,validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)

    print(f"{model_name} Accuracy:", accuracy)

    model.save(save_path)

    print(f"{model_name} saved as {save_path}")


try:
    os.makedirs("trained_models", exist_ok=True)

    # ============ 1. DIGIT CNN MODEL =============

    print("Loading MNIST dataset...")

    mnist_df = load_dataset("mnist_combined.csv")

    y_digits = mnist_df.iloc[:, 0].to_numpy(dtype=np.int64)
    X_digits = mnist_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    train_cnn_model(X_digits,y_digits,num_classes=10,model_name="Digit CNN Model",save_path="trained_models/cnn_digitmodel.keras")

    # ============ 2. LETTER CNN MODEL ============

    print("\nLoading EMNIST Letters dataset...")

    emnist_df = load_dataset("emnist_combined_letters.csv")

    y_letters = emnist_df.iloc[:, 0].to_numpy(dtype=np.int64)
    X_letters = emnist_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    # EMNIST letters: 1-26
    # Convert to: 0-25
    y_letters = y_letters - 1

    train_cnn_model(
        X_letters,
        y_letters,
        num_classes=26,
        model_name="Letter CNN Model",
        save_path="trained_models/cnn_lettermodel.keras"
    )

    print("\nBoth CNN models trained and saved successfully!")

except Exception as e:
    print("Unexpected error:", e)