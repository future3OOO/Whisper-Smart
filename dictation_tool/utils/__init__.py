import logging
import random
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Callable, TypeVar

T = TypeVar("T")

# P3: Logger guard to prevent duplicate handlers
LOGGER = logging.getLogger("dictation_tool")
if not LOGGER.handlers:
    _HANDLER = logging.StreamHandler(sys.stderr)
    _HANDLER.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    )
    LOGGER.addHandler(_HANDLER)
    LOGGER.setLevel("INFO")          # default, overridden by -v / -q


@contextmanager
def timed(label: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        LOGGER.debug("%s took %.1f ms", label, (perf_counter() - start) * 1e3)


def retry(fn: Callable[[], T], attempts: int = 5, backoff: float = 0.05) -> T:
    """Retry helper with exponential back-off and jitter (for fragile clipboard ops)."""

    last_err: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as err:
            last_err = err
            # P3: Add jitter to prevent thundering herd
            base_sleep = backoff * (2**i)
            jitter = random.uniform(0.8, 1.2)  # ±20% jitter
            sleep_time = base_sleep * jitter
            LOGGER.debug("Retry %d in %.2f s due to %s", i + 1, sleep_time, err)
            time.sleep(sleep_time)
    assert last_err is not None
    raise last_err


def clamp_tokens(msg: str, limit: int = 1000) -> str:
    """Clamp message length to keep reasoning tokens budget when used inside agents."""
    if len(msg) <= limit:
        return msg
    return msg[:limit - 3] + "..."
