"""Logging sederhana: satu logger ke konsol + file, dan pencatat tahapan."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def get_logger(name: str = "yolo_seg", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()  # aman dijalankan ulang di Jupyter tanpa log ganda

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    formatter = logging.Formatter(FORMAT, "%H:%M:%S")
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@contextmanager
def stage(logger: logging.Logger, title: str):
    """Catat awal, akhir, durasi, dan kegagalan satu tahapan."""
    logger.info(">> %s", title)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("!! %s GAGAL", title)
        raise
    logger.info("OK %s (%.1f dtk)", title, time.perf_counter() - start)
