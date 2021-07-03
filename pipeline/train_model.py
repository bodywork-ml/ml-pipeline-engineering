"""
- Download dataset.
- Train model.
- Persist model.
"""
from pipeline.utils import configure_logger

log = configure_logger()


def main() -> None:
    log.info("Hello from train_model stage.")


if __name__ == "__main__":
    main()
