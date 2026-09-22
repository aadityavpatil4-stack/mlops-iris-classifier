import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path=".")

df = pd.read_parquet("data/iris_features.parquet")

entity_df = df[["sample_id", "event_timestamp"]].head(5)

historical = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_measurements:sepal length (cm)",
        "iris_measurements:sepal width (cm)",
        "iris_measurements:petal length (cm)",
        "iris_measurements:petal width (cm)",
        "iris_engineered_features:sepal_area",
        "iris_engineered_features:petal_area",
        "iris_engineered_features:sepal_to_petal_length_ratio",
        "iris_engineered_features:petal_length_bin",
    ],
).to_df()

print(historical)
