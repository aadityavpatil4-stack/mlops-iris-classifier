# CI Pipeline

## Overview

The project uses GitHub Actions to compile and lint Python files, run the Practical 8 pytest suite, enforce the model-quality tests, and save test and coverage reports. The workflow is `.github/workflows/ci.yml` and runs for pushes to `main` or `develop` and pull requests targeting either branch.

## Jobs

The `build-and-lint` job sets up Python 3.11, installs the project and CI dependencies, compiles `src` and `tests`, and runs Flake8 on `src`. It ignores only existing whitespace warnings and the unused import in `src/random_search_tuning.py`; those source files were left unchanged.

The `test-suite` job runs after `build-and-lint`. It collects and processes Iris data, runs all tests with pytest coverage, and uploads `test-results.xml` and `coverage.xml` as the `test-and-coverage-reports` artifact.

The `model-quality-gate` job runs after `test-suite`. It prepares the same Iris features and runs `tests/test_model_quality.py`, including the minimum accuracy assertion.

## Verification status

The local Practical 8 suite passed: 20 tests passed. GitHub Actions run `40475682629` failed at lint because it used the initial lint settings. A follow-up commit added the specific ignores described above. The follow-up run and its final result should be checked in the repository's Actions tab.

Branch protection requiring the three job checks and a pull request has not been verified as configured. The deliberate model-regression pull request from the practical has not been run as part of this setup. Configure and verify those GitHub settings before claiming that merges are blocked by CI or that regression blocking has been demonstrated.
