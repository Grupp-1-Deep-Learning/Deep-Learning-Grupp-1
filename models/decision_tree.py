import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def load_emnist_mapping(path):
    mapping = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            label, ascii_code = line.strip().split()
            mapping[int(label)] = chr(int(ascii_code))

    return mapping

print("Laddar EMNIST balanced...")

train_df = pd.read_csv("emnist-balanced-train.csv", header=None)
test_df = pd.read_csv("emnist-balanced-test.csv", header=None)

print("Dataset laddat!")

label_map = load_emnist_mapping("emnist-balanced-mapping.txt")

print("Label mapping:")
print(label_map)

# Första kolumnen = label
X_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, 0]

X_test = test_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("Startar GridSearch...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [20, None],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt"]
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

X_train_small = X_train[:10000]
y_train_small = y_train[:10000]

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    verbose=2,
    n_jobs=1
)

grid_search.fit(X_train_small, y_train_small)

print("Bästa parametrar:")
print(grid_search.best_params_)

model = RandomForestClassifier(
    **grid_search.best_params_,
    random_state=42,
    n_jobs=-1
)

print("Tränar slutmodell på hela träningsdatan...")
model.fit(X_train, y_train)

print("Utvärderar...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

os.makedirs("trained_models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "label_map": label_map
    },
    "trained_models/random_forest_emnist_balanced.joblib"
)

print("Modellen sparad!")