"""Small shared helpers."""

from __future__ import annotations

import contextlib
import fcntl
from pathlib import Path


@contextlib.contextmanager
def directory_lock(path):
    """Hold an exclusive lock on ``<path>/.lock`` (``path`` is created) for the duration of the block.

    Raises:
        ValueError: another process holds the lock.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"another process is writing {path}") from error
        yield


__all__ = ["directory_lock"]
