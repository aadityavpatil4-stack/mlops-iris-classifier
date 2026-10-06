from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "iris-hyperparameter-tuning"

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    print("Experiment not found.")
    exit()

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.cv_f1_macro DESC"]
)

print("\n===== TUNING COMPARISON =====")
print(f"{'Run Name':<25}{'CV F1 Macro':<15}{'Test Accuracy':<15}{'Total Fits':<12}")
print("-" * 67)

for run in runs:
    name = run.data.tags.get("mlflow.runName", "N/A")
    metrics = run.data.metrics
    params = run.data.params

    cv_f1 = metrics.get("best_cv_f1_macro", metrics.get("cv_f1_macro", 0))
    accuracy = metrics.get("test_accuracy", 0)

    search_type = params.get("search_type", "")

    if search_type == "GridSearchCV":
        fits = int(params.get("total_combinations", 72)) * int(params.get("cv_folds", 5))
    elif search_type == "RandomizedSearchCV":
        fits = int(params.get("n_iter", 30)) * int(params.get("cv_folds", 5))
    else:
        fits = 5

    print(f"{name:<25}{cv_f1:<15.4f}{accuracy:<15.4f}{fits:<12}")

print("\nComparison completed successfully.")
