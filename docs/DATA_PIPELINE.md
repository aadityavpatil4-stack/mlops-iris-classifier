# Data Pipeline Documentation

## Purpose

This pipeline automates the complete data workflow for the Iris dataset:

**Collect → Preprocess → Feature Engineering → Validation**

## 1. Data Collection

**Input:** Iris dataset from scikit-learn

**Output:** `data/raw/iris_raw.csv`

The collection stage loads the Iris dataset, converts target values into species names, adds a UTC collection timestamp, and saves the raw dataset as CSV.

## 2. Data Preprocessing

**Input:** `data/raw/iris_raw.csv`

**Output:** `data/processed/iris_preprocessed.csv`

The preprocessing stage removes duplicate rows, converts feature columns to numeric values, fills missing numeric values using median imputation, removes rows with missing target values, and removes the `collected_at` column.

## 3. Feature Engineering

**Input:** `data/processed/iris_preprocessed.csv`

**Output:** `data/processed/iris_features.csv`

The feature engineering stage creates `sepal_area`, `petal_area`, `sepal_to_petal_length_ratio`, and `petal_length_bin`.

## 4. Data Validation

**Input:** `data/processed/iris_features.csv`

The validation stage checks required columns, null values, valid species values, and expected numeric ranges. The pipeline fails if any validation check is unsuccessful.

## Pipeline Flow

```text
Iris Dataset
     |
     v
  Collect
     |
     v
Raw CSV
     |
     v
 Preprocess
     |
     v
Preprocessed CSV
     |
     v
Feature Engineering
     |
     v
Features CSV
     |
     v
  Validate
     |
     v
Validated Dataset
```

## DVC Automation

The pipeline is automated using DVC. The stages defined in `dvc.yaml` are `collect`, `preprocess`, `features`, and `validate`. The `dvc.lock` file records the dependency and output state of the pipeline.
