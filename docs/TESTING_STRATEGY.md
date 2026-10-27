# Experiment 8: Testing Strategy

## Objective

Test data preprocessing, feature engineering, data validation, model training,
predictions, and model quality using pytest.

## Test Files

| Test file | Testing layer | Tests |
|---|---|---:|
| test_preprocess.py | Data processing | 4 |
| test_features.py | Feature engineering | 4 |
| test_validate.py | Data validation | 7 |
| test_training_pipeline.py | Training pipeline | 3 |
| test_model_quality.py | Model quality | 2 |
| **Total** | | **20** |

## Execution

Run the complete suite with coverage and a JUnit XML report:

```bash
pytest --cov=src --cov-report=term-missing --junitxml=test-results.xml
```

## Final Test Results

- Tests passed: 20
- Tests failed: 0
- Warnings: 0
- Overall source-code coverage: 26% (352 statements, 259 missed)
- JUnit XML report: `test-results.xml`
- Model-quality gate: accuracy 0.9667 against the required minimum of 0.90

## Coverage by Pipeline Module

| Module | Statements | Coverage |
|---|---:|---:|
| preprocess.py | 29 | 100% |
| features.py | 20 | 100% |
| validate.py | 44 | 100% |

The overall percentage includes source files outside these tests: `collect.py`,
training entry points, and hyperparameter tuning scripts. Those modules remain
untested by this Experiment 8 suite. The three pipeline modules exercised here
are fully covered, including their command-line entry points and validation
error paths.

## Conclusion

All 20 tests passed with no warnings. The tests cover preprocessing, feature
engineering, data validation, model training and prediction behavior, and the
0.90 minimum-accuracy quality gate. A separate smoke check raised the threshold
to 1.01 and confirmed the gate rejects the measured 0.9667 accuracy; the
required 0.90 threshold was restored afterward.
