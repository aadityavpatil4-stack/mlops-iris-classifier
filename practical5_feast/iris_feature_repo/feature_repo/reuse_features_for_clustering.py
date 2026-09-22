from feast import FeatureStore

store = FeatureStore(repo_path=".")

feature_service = store.get_feature_service("iris_feature_service")

features = store.get_online_features(
    features=feature_service,
    entity_rows=[{"sample_id": 1}],
).to_dict()

for key, value in features.items():
    print(f"{key}: {value}")
