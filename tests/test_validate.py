import pandas as pd
import pytest

from src.pipeline.validate import validate, DataValidationError


VALID_ROW = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
    "species": "setosa",
    "sepal_area": 17.85,
    "petal_area": 0.28,
    "sepal_to_petal_length_ratio": 3.64,
    "petal_length_bin": "short",
}


def write_csv(tmp_path, df):
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_validate_passes_on_clean_data(tmp_path):
    df = pd.DataFrame([VALID_ROW, VALID_ROW])
    path = write_csv(tmp_path, df)

    result = validate(path)

    assert len(result) == 2


def test_validate_fails_on_missing_column(tmp_path):
    df = pd.DataFrame([VALID_ROW]).drop(columns=["sepal_area"])
    path = write_csv(tmp_path, df)

    with pytest.raises(
        DataValidationError, match="Missing expected columns"
    ):
        validate(path)


def test_validate_fails_on_invalid_species(tmp_path):
    row = VALID_ROW.copy()
    row["species"] = "unicorn"
    path = write_csv(tmp_path, pd.DataFrame([row]))

    with pytest.raises(
        DataValidationError, match="Unexpected species values"
    ):
        validate(path)


def test_validate_fails_on_out_of_range_value(tmp_path):
    row = VALID_ROW.copy()
    row["sepal length (cm)"] = 50.0
    path = write_csv(tmp_path, pd.DataFrame([row]))

    with pytest.raises(
        DataValidationError, match="out of expected range"
    ):
        validate(path)


def test_validate_reports_null_values_and_missing_species_safely(tmp_path):
    row = VALID_ROW.copy()
    row.pop("species")
    row["sepal length (cm)"] = None
    path = write_csv(tmp_path, pd.DataFrame([row]))

    with pytest.raises(DataValidationError) as exc_info:
        validate(path)

    assert "Missing expected columns" in str(exc_info.value)
    assert "Unexpected null values" in str(exc_info.value)


def test_validate_cli_success_and_failure(tmp_path, monkeypatch):
    import runpy
    import sys

    valid_path = write_csv(tmp_path, pd.DataFrame([VALID_ROW]))
    monkeypatch.setattr(sys, "argv", ["validate", "--input", valid_path])
    monkeypatch.delitem(sys.modules, "src.pipeline.validate", raising=False)
    runpy.run_module("src.pipeline.validate", run_name="__main__")

    bad_row = VALID_ROW.copy()
    bad_row["species"] = "unicorn"
    invalid_path = write_csv(tmp_path, pd.DataFrame([bad_row]))
    monkeypatch.setattr(sys, "argv", ["validate", "--input", invalid_path])
    with pytest.raises(SystemExit) as exc_info:
        monkeypatch.delitem(sys.modules, "src.pipeline.validate", raising=False)
        runpy.run_module("src.pipeline.validate", run_name="__main__")
    assert exc_info.value.code == 1


def test_validate_skips_range_checks_for_absent_columns(tmp_path):
    row = VALID_ROW.copy()
    row.pop("sepal length (cm)")
    path = write_csv(tmp_path, pd.DataFrame([row]))

    with pytest.raises(DataValidationError, match="Missing expected columns"):
        validate(path)
