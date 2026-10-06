import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import randint

import mlflow
import mlflow.sklearn


# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "data/processed/iris_features.csv"

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

X = df[FEATURES].copy()
y = df["species"].copy()

# Handle missing values
X = X.fillna(X.median())

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Random Search parameters
# -----------------------------
PARAM_DIST = {
    "n_estimators": randint(50, 300),
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": randint(2, 15),
    "max_features": ["sqrt", "log2"],
}

N_ITER = 30
CV_FOLDS = 5


# -----------------------------
# Randomized Search
# -----------------------------
rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=PARAM_DIST,
    n_iter=N_ITER,
    cv=CV_FOLDS,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1,
    return_train_score=False
)

random_search.fit(X_train, y_train)


# -----------------------------
# Best model evaluation
# -----------------------------
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

best_cv_score = random_search.best_score_


# -----------------------------
# MLflow tracking
# -----------------------------
mlflow.set_experiment("iris-hyperparameter-tuning")

with mlflow.start_run(run_name="random_search"):

    mlflow.log_param("search_type", "RandomizedSearchCV")
    mlflow.log_param("n_iter", N_ITER)
    mlflow.log_param("cv_folds", CV_FOLDS)

    mlflow.log_metric("best_cv_f1_macro", best_cv_score)
    mlflow.log_metric("test_accuracy", test_accuracy)

    for param_name, param_value in random_search.best_params_.items():
        mlflow.log_param(f"best_{param_name}", param_value)

    # Save all search candidates
    results = pd.DataFrame(random_search.cv_results_)

    candidate_columns = [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]

    results[candidate_columns].to_csv(
        "random_search_all_candidates.csv",
        index=False
    )

    mlflow.log_artifact("random_search_all_candidates.csv")

    mlflow.sklearn.log_model(
        best_model,
        "random_forest_random_search"
    )


# -----------------------------
# Results
# -----------------------------
print(
    f"Random Search evaluated {N_ITER} combinations x "
    f"{CV_FOLDS} folds = {N_ITER * CV_FOLDS} total fits"
)

print("Best params:", random_search.best_params_)
print(f"Best CV f1_macro: {best_cv_score:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
