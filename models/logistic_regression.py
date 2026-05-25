# TWO LOGISTIC REGRESSION MODELS
# 1. DIGIT MODEL  -> MNIST
# 2. LETTER MODEL -> EMNIST LETTERS

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_dataset(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"{file_path} is empty")
        if df.shape[1] < 785:
            raise ValueError(f"{file_path} does not have 785 columns")
        return df
    except Exception as e:
        print(f"Dataset loading error: {e}")

def train_model(X, y, model_name):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        model = LogisticRegression(max_iter=100,solver="lbfgs",n_jobs=-1,random_state=42)
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy:", accuracy)
        return model
    except Exception as e:
        print(f"Training error in {model_name}:", e)

try:
    os.makedirs("trained_models", exist_ok=True)

    # ====== 1. DIGIT MODEL ======
    print("Loading MNIST dataset...")

    mnist_df = load_dataset("mnist_combined.csv")

    y_digits = mnist_df.iloc[:, 0].to_numpy(dtype=np.int64)
    X_digits = mnist_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    X_digits = X_digits / 255.0

    digit_model =train_model(X_digits, y_digits, "Digit Model")

    joblib.dump(digit_model,"trained_models/logistic_digitmodel.joblib")

    print("Digit model saved")

    # ======= 2. LETTER MODEL =======
    print("\nLoading EMNIST Letters dataset...")

    emnist_df = load_dataset("emnist_combined_letters.csv")

    y_letters = emnist_df.iloc[:, 0].to_numpy(dtype=np.int64)
    X_letters = emnist_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    # EMNIST labels are 1-26
    # Convert to 0-25 so app can display A-Z
    y_letters = y_letters - 1

    X_letters = X_letters / 255.0

    letter_model = train_model(X_letters, y_letters, "Letter Model")

    joblib.dump(letter_model,"trained_models/logistic_lettermodel.joblib")

    print("Letter model saved as trained_models/logistic_lettermodel.joblib")
except Exception as e:
    print("Unexpected error:", e)
