"""
Tests for web API.
"""
import pickle
from subprocess import run
from unittest.mock import patch

from bodywork_pipeline_utils.aws import Model
from fastapi.testclient import TestClient
from numpy import array

from pipeline.serve_model import app

test_client = TestClient(app)


def wrapped_model() -> Model:
    with open("tests/resources/model.pkl", "r+b") as file:
        wrapped_model = pickle.load(file)
    return wrapped_model


@patch("pipeline.serve_model.wrapped_model", new=wrapped_model(), create=True)
def test_web_api_returns_valid_response_given_valid_data():
    prediction_request = {"product_code": "SKU001", "orders_placed": 100}
    prediction_response = test_client.post(
        "/api/v0.1/time_to_dispatch", json=prediction_request
    )
    model_obj = wrapped_model()
    expected_prediction = model_obj.model.predict(array([[100, 0]])).tolist()[0]
    assert prediction_response.status_code == 200
    assert prediction_response.json()["est_hours_to_dispatch"] == expected_prediction
    assert prediction_response.json()["model_version"] == str(model_obj)


@patch("pipeline.serve_model.wrapped_model", new=wrapped_model(), create=True)
def test_web_api_returns_error_code_given_invalid_data():
    prediction_request = {"product_code": "SKU001", "foo": 100}
    prediction_response = test_client.post(
        "/api/v0.1/time_to_dispatch", json=prediction_request
    )
    assert prediction_response.status_code == 422
    assert "value_error.missing" in prediction_response.text

    prediction_request = {"product_code": "SKU000", "orders_placed": 100}
    prediction_response = test_client.post(
        "/api/v0.1/time_to_dispatch", json=prediction_request
    )
    assert prediction_response.status_code == 422
    assert "not a valid enumeration member" in prediction_response.text

    prediction_request = {"product_code": "SKU001", "orders_placed": -100}
    prediction_response = test_client.post(
        "/api/v0.1/time_to_dispatch", json=prediction_request
    )
    assert prediction_response.status_code == 422
    assert "ensure this value is greater than or equal to 0" in prediction_response.text


def test_web_server_raises_exception_if_passed_invalid_args():
    process = run(
        ["python", "-m", "pipeline.serve_model"], capture_output=True, encoding="utf-8"
    )
    assert process.returncode != 0
    assert "ERROR" in process.stdout
    assert "Invalid arguments passed to serve_model.py" in process.stdout
