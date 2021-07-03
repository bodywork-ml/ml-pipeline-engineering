"""
Tests for model training stage.
"""
from _pytest.logging import LogCaptureFixture

from pipeline.train_model import main


def test_main_execution(caplog: LogCaptureFixture):
    main()
    logs = caplog.text
    assert "Hello from train_model stage" in logs
