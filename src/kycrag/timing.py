"""Step-level timing logs for the retrieval/generation pipeline."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("kycrag.timing")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@contextmanager
def timer(step: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%-16s %8.1fms", step, elapsed_ms)
