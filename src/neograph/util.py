"""Logging, batching, hashing helpers."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Iterator
from typing import TypeVar

from rich.console import Console
from rich.logging import RichHandler

T = TypeVar("T")

_console = Console()


def get_logger(name: str = "neograph") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RichHandler(console=_console, rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def chunk(it: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive lists of length `size` from iterable `it`."""
    buf: list[T] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def sha1_short(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def exit_marker(name: str, ok: bool, **details: object) -> None:
    """Print a one-line PASS/FAIL marker greppable by phase scripts."""
    label = "PASS" if ok else "FAIL"
    extras = " ".join(f"{k}={v}" for k, v in details.items())
    print(f"[exit-criterion: {label}] {name} {extras}")
