import pandas as pd
import numpy as np
import os

from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

TRAIN_FILE = "emnist-byclass-train.csv"
TEST_FILE = "emnist-byclass-test.csv"
MODEL_SAVE_PATH = "trained_models/cnn_combined_model.keras"
NUM_CLASSES = 62

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return X, y

def fix_emnist_orientation(X):
    X= X.reshape(-1, 28, 28)
    fixed_images = []
    for img in X:
        pil_img = Image.fromarray(img.astype(np.uint8))
        pil_img = pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        pil_img = pil_img.rotate(90, expand=False)
        fixed_images.append(np.array(pil_img))
    return np.array(fixed_images).reshape(-1, 784)

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),

        Flatten(),

        Dense(256, activation="relu"),
        Dropout(0.3),

        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    return model


os.makedirs("trained_models", exist_ok=True)

print("Loading EMNIST ByClass train dataset...")
X_train, y_train = load_dataset(TRAIN_FILE)

print("Loading EMNIST ByClass test dataset...")
X_test, y_test = load_dataset(TEST_FILE)

print("Fixing EMNIST orientation...")
X_train = fix_emnist_orientation(X_train)
X_test = fix_emnist_orientation(X_test)

print("Normalizing images...")
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Reshaping images for CNN...")
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("Building CNN model...")

model = build_cnn_model()

print("Training CNN ByClass model...")
model.fit(X_train,y_train,epochs=10,batch_size=128,validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print("Combined CNN accuracy:", accuracy)

model.save(MODEL_SAVE_PATH)
print(f"Saved as {MODEL_SAVE_PATH}")