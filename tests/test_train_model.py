"""
Tests for model training stage.
"""
from datetime import datetime
from subprocess import run
from unittest.mock import MagicMock, patch

from bodywork_pipeline_utils.aws import Dataset
from pandas import read_csv, DataFrame
from pytest import fixture, raises
from _pytest.logging import LogCaptureFixture
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pipeline.train_model import (
    FeatureAndLabels,
    main,
    prepare_data,
    preprocess,
    train_model,
    validate_trained_model_logic,
)


@fixture(scope="session")
def dataset() -> Dataset:
    data = read_csv("tests/resources/dataset.csv")
    dataset = Dataset(data, datetime(2021, 7, 15), "tests", "resources", "foobar")
    return dataset


@fixture(scope="session")
def prepared_data(dataset: Dataset) -> FeatureAndLabels:
    return FeatureAndLabels(
        dataset.data[["orders_placed", "product_code"]][:800],
        dataset.data[["orders_placed", "product_code"]][800:999],
        dataset.data["hours_to_dispatch"][:800],
        dataset.data["hours_to_dispatch"][800:999],
    )


def test_prepare_data_splits_labels_and_features_into_test_and_train(dataset: Dataset):
    label_column = "hours_to_dispatch"
    n_rows_in_dataset = dataset.data.shape[0]
    n_cols_in_dataset = dataset.data.shape[1]
    prepared_data = prepare_data(dataset.data)

    assert prepared_data.X_train.shape[1] == n_cols_in_dataset - 1
    assert label_column not in prepared_data.X_train.columns

    assert prepared_data.X_test.shape[1] == n_cols_in_dataset - 1
    assert label_column not in prepared_data.X_test.columns

    assert prepared_data.y_train.ndim == 1
    assert prepared_data.y_train.name == label_column

    assert prepared_data.y_test.ndim == 1
    assert prepared_data.y_test.name == label_column

    assert (
        prepared_data.X_train.shape[0] + prepared_data.X_test.shape[0]
        == n_rows_in_dataset
    )

    assert (
        prepared_data.y_train.shape[0] + prepared_data.y_test.shape[0]
        == n_rows_in_dataset
    )


def test_preprocess_processes_features():
    data = DataFrame({"orders_placed": [30], "product_code": ["SKU004"]})
    processed_data = preprocess(data)
    assert processed_data[0, 0] == 30
    assert processed_data[0, 1] == 3


def test_train_model_yields_model_and_metrics(prepared_data: FeatureAndLabels):
    model, metrics = train_model(prepared_data, {"random_state": [42]})
    try:
        check_is_fitted(model)
        assert True
    except NotFittedError:
        assert False

    assert metrics.r_squared >= 0.9
    assert metrics.mean_absolute_error <= 1.25


def test_validate_trained_model_logic_raises_exception_for_failing_models(
    prepared_data: FeatureAndLabels,
):
    dummy_model = DummyRegressor(strategy="constant", constant=-1.0)
    dummy_model.fit(prepared_data.X_train, prepared_data.y_train)
    expected_exception_str = (
        "Trained model failed verification: "
        "hours_to_dispatch predictions do not increase with orders_placed."
    )
    with raises(RuntimeError, match=expected_exception_str):
        validate_trained_model_logic(dummy_model, prepared_data)

    dummy_model = DummyRegressor(strategy="constant", constant=-1.0)
    dummy_model.fit(prepared_data.X_train, prepared_data.y_train)
    expected_exception_str = (
        "Trained model failed verification: "
        "hours_to_dispatch predictions do not increase with orders_placed, "
        "negative hours_to_dispatch predictions found for test set."
    )
    with raises(RuntimeError, match=expected_exception_str):
        validate_trained_model_logic(dummy_model, prepared_data)

    dummy_model = DummyRegressor(strategy="constant", constant=1000.0)
    dummy_model.fit(prepared_data.X_train, prepared_data.y_train)
    expected_exception_str = (
        "Trained model failed verification: "
        "hours_to_dispatch predictions do not increase with orders_placed, "
        "outlier hours_to_dispatch predictions found for test set."
    )
    with raises(RuntimeError, match=expected_exception_str):
        validate_trained_model_logic(dummy_model, prepared_data)


@patch("pipeline.train_model.aws")
def test_train_job_happy_path(
    mock_aws: MagicMock,
    dataset: Dataset,
    caplog: LogCaptureFixture,
):
    mock_aws.get_latest_csv_dataset_from_s3.return_value = dataset
    main("project-bucket", 0.8, 0.9, {"random_state": [42]})
    mock_aws.Model().put_model_to_s3.assert_called_once()
    logs = caplog.text
    assert "Starting train-model stage" in logs
    assert "Retrieved dataset from s3" in logs
    assert "Trained model" in logs
    assert "Model serialised and persisted to s3" in logs


@patch("pipeline.train_model.aws")
def test_train_job_raises_exception_when_metrics_below_error_threshold(
    mock_aws: MagicMock,
    dataset: Dataset,
):
    mock_aws.get_latest_csv_dataset_from_s3.return_value = dataset
    with raises(RuntimeError, match="below deployment threshold"):
        main("project-bucket", 1, 0.9, {"random_state": [42]})


@patch("pipeline.train_model.aws")
def test_train_job_logs_warning_when_metrics_below_warning_threshold(
    mock_aws: MagicMock,
    dataset: Dataset,
    caplog: LogCaptureFixture,
):
    mock_aws.get_latest_csv_dataset_from_s3.return_value = dataset
    main("project-bucket", 0.5, 0.9, {"random_state": [42]})
    assert "WARNING" in caplog.text
    assert "breached warning threshold" in caplog.text


def test_run_job_handles_error_for_invalid_args():
    process_one = run(
        ["python", "pipeline/train_model.py"], capture_output=True, encoding="utf-8"
    )
    assert process_one.returncode != 0
    assert "ERROR" in process_one.stdout
    assert "Invalid arguments passed to train_model.py" in process_one.stdout

    process_two = run(
        ["python", "-m", "pipeline.train_model", "my-bucket", "-1", "0.5"],
        capture_output=True,
        encoding="utf-8",
    )
    assert process_two.returncode != 0
    assert "ERROR" in process_two.stdout
    assert "Invalid arguments passed to train_model.py" in process_two.stdout

    process_three = run(
        ["python", "-m", "pipeline.train_model", "my-bucket", "2", "0.5"],
        capture_output=True,
        encoding="utf-8",
    )
    assert process_three.returncode != 0
    assert "ERROR" in process_three.stdout
    assert "Invalid arguments passed to train_model.py" in process_three.stdout

    process_four = run(
        ["python", "-m", "pipeline.train_model", "my-bucket", "0.5", "-1"],
        capture_output=True,
        encoding="utf-8",
    )
    assert process_four.returncode != 0
    assert "ERROR" in process_four.stdout
    assert "Invalid arguments passed to train_model.py" in process_four.stdout

    process_five = run(
        ["python", "-m", "pipeline.train_model", "my-bucket", "0.5", "2"],
        capture_output=True,
        encoding="utf-8",
    )
    assert process_five.returncode != 0
    assert "ERROR" in process_five.stdout
    assert "Invalid arguments passed to train_model.py" in process_five.stdout
