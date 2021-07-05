"""
Tests for web API.
"""
from fastapi.testclient import TestClient

from pipeline.serve_model import app

test_client = TestClient(app)


def test_web_api_returns_valid_response_given_valid_data():
    prediction_request = {"product_code": "SKU001", "orders_placed": 100}
    prediction_response = test_client.post(
        "/api/v0.0.1/time_to_dispatch", json=prediction_request
    )
    assert prediction_response.status_code == 200
    assert "est_hours_to_dispatch" in prediction_response.json().keys()
    assert "model_version" in prediction_response.json().keys()


def test_web_api_returns_error_code_given_invalid_data():
    prediction_request = {"product_code": "SKU001", "orders": 100}
    prediction_response = test_client.post(
        "/api/v0.0.1/time_to_dispatch", json=prediction_request
    )
    assert prediction_response.status_code == 422
    assert "value_error.missing" in prediction_response.text
