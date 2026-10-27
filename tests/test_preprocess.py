import pandas as pd
import pytest

from src.pipeline.preprocess import preprocess

NUMERIC_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture
def valid_raw_df(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.1, 4.9, 4.7],
        "sepal width (cm)": [3.5, 3.0, 3.2],
        "petal length (cm)": [1.4, 1.4, 1.3],
        "petal width (cm)": [0.2, 0.2, 0.2],
        "species": ["setosa", "setosa", "setosa"],
        "collected_at": ["2026-01-01"] * 3,
    })

    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


def test_preprocess_removes_duplicates(tmp_path, valid_raw_df):
    df = pd.read_csv(valid_raw_df)
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    input_path = tmp_path / "raw_with_dupe.csv"
    output_path = tmp_path / "out.csv"
    df_with_dupe.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 3


def test_preprocess_imputes_missing_numeric_values(
    tmp_path, valid_raw_df
):
    df = pd.read_csv(valid_raw_df)
    df.loc[0, "sepal length (cm)"] = None

    input_path = tmp_path / "raw_with_missing.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert result["sepal length (cm)"].isnull().sum() == 0
    assert result.loc[0, "sepal length (cm)"] == pytest.approx(
        df["sepal length (cm)"].median()
    )


def test_preprocess_drops_rows_with_missing_target(
    tmp_path, valid_raw_df
):
    df = pd.read_csv(valid_raw_df)
    df.loc[0, "species"] = None

    input_path = tmp_path / "raw_missing_target.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 2


def test_preprocess_cli_runs_with_explicit_paths(tmp_path, monkeypatch):
    import runpy
    import sys

    source = pd.DataFrame({
        "sepal length (cm)": [5.1],
        "sepal width (cm)": [3.5],
        "petal length (cm)": [1.4],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
    })
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "processed.csv"
    source.to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys, "argv", ["preprocess", "--input", str(input_path), "--output", str(output_path)]
    )

    monkeypatch.delitem(sys.modules, "src.pipeline.preprocess", raising=False)
    runpy.run_module("src.pipeline.preprocess", run_name="__main__")

    assert output_path.exists()
