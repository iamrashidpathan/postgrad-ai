"""
utils.py

Small shared helpers used across the project.

Demonstrates: logging setup, kept in one place so every script configures
logging identically.
"""

import logging
from pathlib import Path


def setup_logging(log_path: str = "logs/train.log") -> logging.Logger:
    """
    Configure a logger that writes timestamped messages to a log file.

    Using a named logger (rather than the root logger) is good practice:
    it avoids clashing with logging config from other libraries.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ai_mini_project")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate log lines if called twice

    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    return logger
