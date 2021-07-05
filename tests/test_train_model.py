"""
Tests for model training stage.
"""
from _pytest.logging import LogCaptureFixture

from pipeline.train_model import main


def test_main_logs_info_msg(caplog: LogCaptureFixture):
    main()
    logs = caplog.text
    assert "Starting train-model stage" in logs
