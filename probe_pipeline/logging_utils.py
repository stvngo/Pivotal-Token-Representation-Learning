"""Logging setup helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def build_logger(name: str, log_path: str | Path, level: str = "INFO") -> logging.Logger:
    """Create a logger that writes to both stdout and file."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

