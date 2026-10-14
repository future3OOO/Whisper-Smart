from __future__ import annotations

import json
import pathlib
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

_lock = threading.Lock()
_now = time.perf_counter


@contextmanager
def prof(label: str, path: str | None) -> Iterator[None]:
    """Write {ts, label, ms} to JSONL path (if given)."""
    if not path:
        yield
        return
    t0 = _now()
    try:
        yield
    finally:
        dur = (_now() - t0) * 1_000
        rec = {"ts": time.time(), "label": label, "ms": round(dur, 3)}
        with _lock:
            pathlib.Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
