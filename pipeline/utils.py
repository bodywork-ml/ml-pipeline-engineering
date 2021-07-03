"""
Utility functions that are common between stages.
"""
import sys
import logging


def configure_logger() -> logging.Logger:
    """Configure a logger that will write to stdout."""
    log_handler = logging.StreamHandler(sys.stdout)
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s"
    )
    log_handler.setFormatter(log_format)
    log = logging.getLogger(__name__)
    log.addHandler(log_handler)
    log.setLevel(logging.INFO)
    return log
