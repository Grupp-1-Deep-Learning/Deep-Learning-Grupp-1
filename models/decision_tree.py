import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("Laddar dataset...")

df = pd.read_csv("mnist_combined.csv")

print("Dataset laddat!")

# Features och labels
X = df.drop("label", axis=1)
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


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

# Spara modell
os.makedirs("trained_models", exist_ok=True)

joblib.dump(
    model,
    "trained_models/random_forest_mnist.joblib"
)

print("Modellen sparad!")