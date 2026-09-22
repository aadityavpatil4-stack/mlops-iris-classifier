# Feature Store Analysis

## Benefits of Using a Feature Store

### 1. Eliminates Training-Serving Skew
The same feature definitions can be reused for historical training data and online serving, reducing differences between training and production features.

### 2. Reusability
Feature definitions and feature services can be reused across multiple ML workflows without recreating feature engineering logic.

### 3. Centralized Feature Governance
A feature store provides a centralized place to manage, document, and retrieve features consistently.

## Experiment 5 Summary

The Feast feature repository contains:
- Entity: `sample_id`
- FeatureView: `iris_measurements`
- FeatureView: `iris_engineered_features`
- FeatureService: `iris_feature_service`
- Online store: SQLite
- Offline source: Parquet

Both online and historical feature retrieval were successfully demonstrated.
