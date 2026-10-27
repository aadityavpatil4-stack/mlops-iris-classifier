import pandas as pd

from src.pipeline.features import engineer_features


def test_sepal_and_petal_area_computed_correctly(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [4.0],
        "petal length (cm)": [2.0],
        "petal width (cm)": [1.0],
        "species": ["setosa"],
    })

    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = engineer_features(str(input_path), str(output_path))

    assert result.loc[0, "sepal_area"] == 20.0
    assert result.loc[0, "petal_area"] == 2.0
    assert result.loc[0, "sepal_to_petal_length_ratio"] == 2.5


def test_petal_length_bin_assigns_expected_category(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0, 5.0, 5.0],
        "sepal width (cm)": [3.0, 3.0, 3.0],
        "petal length (cm)": [1.0, 3.0, 6.0],
        "petal width (cm)": [0.2, 1.0, 2.0],
        "species": ["setosa", "versicolor", "virginica"],
    })

    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = engineer_features(str(input_path), str(output_path))

    assert list(result["petal_length_bin"]) == [
        "short", "medium", "long"
    ]


def test_zero_petal_length_does_not_crash(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [0.0],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
    })

    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = engineer_features(str(input_path), str(output_path))

    assert pd.isna(
        result.loc[0, "sepal_to_petal_length_ratio"]
    )


def test_features_cli_runs_with_explicit_paths(tmp_path, monkeypatch):
    import runpy
    import sys

    source = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [2.0],
        "petal width (cm)": [1.0],
        "species": ["setosa"],
    })
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "features.csv"
    source.to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys, "argv", ["features", "--input", str(input_path), "--output", str(output_path)]
    )

    monkeypatch.delitem(sys.modules, "src.pipeline.features", raising=False)
    runpy.run_module("src.pipeline.features", run_name="__main__")

    assert output_path.exists()
