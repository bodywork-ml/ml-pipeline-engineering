"""
Tests for model training stage.
"""
from _pytest.capture import CaptureFixture

from pipeline.train_model import main


def test_main_execution(capsys: CaptureFixture):
    main()
    stdout = capsys.readouterr().out
    assert "Hello from train_model stage" in stdout
