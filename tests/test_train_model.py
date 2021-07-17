"""
Tests for model training stage.
"""
from datetime import datetime
from subprocess import run
from unittest.mock import MagicMock, patch

from numpy import array
from bodywork_pipeline_utils.aws import Dataset
from pandas import read_csv, DataFrame
from pytest import approx, fixture, raises
from _pytest.logging import LogCaptureFixture
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pipeline.train_model import (
    compute_metrics,
    FeatureAndLabels,
    main,
    prepare_data,
    preprocess,
    train_model
)


@fixture(scope="session")
def dataset() -> Dataset:
    data = read_csv("tests/resources/dataset.csv")
    dataset = Dataset(data, datetime(2021, 7, 15), "tests", "resources", "foobar")
    return dataset


def test_prepare_data_splits_labels_and_features_into_test_and_train(dataset: Dataset):
    label_column = "hours_to_dispatch"
    n_rows_in_dataset = dataset.data.shape[0]
    prepared_data = prepare_data(dataset.data)
    assert prepared_data.X_train.shape[0] == approx(0.8 * n_rows_in_dataset)
    assert prepared_data.X_train.shape[1] == 2
    assert label_column not in prepared_data.X_train.columns
    
    assert prepared_data.X_test.shape[0] == approx(0.2 * n_rows_in_dataset)
    assert prepared_data.X_test.shape[1] == 2
    assert label_column not in prepared_data.X_test.columns

    assert prepared_data.y_train.ndim == 1
    assert prepared_data.y_train.name == label_column
    assert prepared_data.y_train.shape[0] == approx(0.8 * n_rows_in_dataset)

    assert prepared_data.y_test.ndim == 1
    assert prepared_data.y_test.name == label_column
    assert prepared_data.y_test.shape[0] == approx(0.2 * n_rows_in_dataset)


def test_preprocess_processes_features(dataset: Dataset):
    data = DataFrame({"orders_placed": [30], "product_code": ["SKU004"]})
    processed_data = preprocess(data)
    assert processed_data[0, 0] == 30
    assert processed_data[0, 1] == 3


def test_compute_metrics():
    y_actual = array([5, 10, 15])
    y_pred = array([4, 11, 14])
    metrics = compute_metrics(y_actual, y_pred)
    assert metrics.r_squared == 0.94
    assert metrics.mean_absolute_error == 1.0


def test_train_model_yields_valid_model(dataset: Dataset):
    prepared_data = FeatureAndLabels(
        dataset.data[["orders_placed", "product_code"]][:10],
        dataset.data[["orders_placed", "product_code"]][10:20],
        dataset.data["hours_to_dispatch"][:10],
        dataset.data["hours_to_dispatch"][10:20]
    )
    model, metrics = train_model(prepared_data, {"random_state": [42]})
    try:
        check_is_fitted(model)
        assert True
    except NotFittedError:
        assert False
    prediction = model.predict(array([[12, 3]])).tolist()[0]
    assert prediction > 0


@patch("pipeline.train_model.aws")
def test_train_job_happy_path(
    mock_aws: MagicMock,
    dataset: Dataset,
    caplog: LogCaptureFixture,
):
    mock_aws.get_latest_csv_dataset_from_s3.return_value = dataset
    main("project-bucket", 0.8, {"random_state": [42]})
    mock_aws.Model().put_model_to_s3.assert_called_once()
    logs = caplog.text
    assert "Starting train-model stage" in logs
    assert "Retrieved dataset from s3" in logs
    assert "Trained model" in logs
    assert "Model serialised and persisted to s3" in logs


@patch("pipeline.train_model.aws")
def test_train_job_raises_exception_when_metrics_below_threshold(
    mock_aws: MagicMock,
    dataset: Dataset,
):
    mock_aws.get_latest_csv_dataset_from_s3.return_value = dataset
    with raises(RuntimeError, match="below deployment threshold"):
        main("project-bucket", 1, {"random_state": [42]})


def test_run_job_handles_error_for_invalid_args():
    process_one = run(
        ["python", "pipeline/train_model.py"], capture_output=True, encoding="utf-8"
    )
    assert process_one.returncode != 0
    assert "ERROR" in process_one.stdout

    process_two = run(
        ["python", "pipeline/train_model.py", "my-bucket", "-1"],
        capture_output=True,
        encoding="utf-8"
    )
    assert process_two.returncode != 0
    assert "ERROR" in process_two.stdout

    process_three = run(
        ["python", "pipeline/train_model.py", "my-bucket", "2"],
        capture_output=True,
        encoding="utf-8"
    )
    assert process_three.returncode != 0
    assert "ERROR" in process_three.stdout
