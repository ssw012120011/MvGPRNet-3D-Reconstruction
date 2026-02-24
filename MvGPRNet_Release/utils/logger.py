"""Logging utilities."""
import logging
import os


def setup_logging(run_dir: str) -> None:
    """Configure logging to file and console."""
    log_file = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )


def setup_inference_logging(output_dir: str) -> None:
    """Configure inference logging."""
    log_file = os.path.join(output_dir, "inference.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )
