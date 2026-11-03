import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/processed/iris_features.csv"
MODEL_DIR = "models"

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

TARGET_COLUMN = "species"


def train_model():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X, y_encoded)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/final_model.joblib")
    joblib.dump(encoder, f"{MODEL_DIR}/label_encoder.joblib")

    print("Model training completed!")
    print("Classes:", list(encoder.classes_))
    print("Saved model files in:", MODEL_DIR)


if __name__ == "__main__":
    train_model()
