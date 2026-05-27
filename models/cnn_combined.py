import pandas as pd
import numpy as np
import os

from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return X, y

def fix_emnist_letters_orientation(X_letters):
    X_letters = X_letters.reshape(-1, 28, 28)
    fixed_images = []
    for img in X_letters:
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

        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(36, activation="softmax")
    ])

    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    return model


os.makedirs("trained_models", exist_ok=True)

print("Loading MNIST digits...")
X_digits, y_digits = load_dataset("mnist_combined.csv")

print("Loading EMNIST letters...")
X_letters, y_letters = load_dataset("emnist_combined_letters.csv")

print("Fixing EMNIST letter orientation...")
X_letters = fix_emnist_letters_orientation(X_letters)

# Digits: 0-9
# EMNIST letters: 1-26
# Convert letters to 10-35
y_letters = y_letters - 1
y_letters = y_letters + 10

print("Combining datasets...")
X = np.vstack([X_digits, X_letters])
y = np.concatenate([y_digits, y_letters])

X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = build_cnn_model()

print("Training combined CNN...")
model.fit(X_train,y_train,epochs=10,batch_size=128,validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print("Combined CNN accuracy:", accuracy)

model.save("trained_models/cnn_combined_model.keras")
print("Saved as trained_models/cnn_combined_model.keras")