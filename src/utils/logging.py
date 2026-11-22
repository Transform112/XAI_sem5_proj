"""Small logging helper shared by every stage.

Stages log to stdout *and* optionally to a run log file under
``runs/logs/<stage>.log``; the file disappears from git (``*.log`` is ignored)
but lets a failed background job be diagnosed after the terminal is gone.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"
_HANDLER_MARK = "_xdrift_handler"


def setup_logging(
    name: str = "xdrift",
    *,
    level: str | int = "INFO",
    log_path: str | Path | None = None,
    stream: Any | None = None,
) -> logging.Logger:
    """Configure and return a logger (idempotent: repeat calls add no duplicates)."""
    logger = logging.getLogger(name)
    logger.setLevel(level if isinstance(level, int) else str(level).upper())
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream_handler = logging.StreamHandler(stream or sys.stdout)
    stream_handler.setFormatter(formatter)
    setattr(stream_handler, _HANDLER_MARK, True)
    logger.addHandler(stream_handler)

    if log_path is not None:
        target = Path(log_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARK, True)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "xdrift") -> logging.Logger:
    """Return a logger, creating a default stdout-only configuration if needed."""
    logger = logging.getLogger(name)
    if not any(getattr(handler, _HANDLER_MARK, False) for handler in logger.handlers):
        return setup_logging(name)
    return logger


__all__ = ["get_logger", "setup_logging"]
