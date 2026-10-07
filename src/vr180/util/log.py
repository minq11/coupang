"""Stage 실행 중 로그 / 진행률 / 취소 — CLI 와 GUI 가 같은 인터페이스를 쓴다."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable


class Cancelled(Exception):
    pass


class RunContext:
    def __init__(
        self,
        log_fn: Callable[[str], None] | None = None,
        progress_fn: Callable[[float, str], None] | None = None,
    ):
        self._log_fn = log_fn or (lambda s: print(s, file=sys.stderr, flush=True))
        self._progress_fn = progress_fn or (lambda f, s: None)
        self._cancel = threading.Event()
        self.lines: list[str] = []

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.lines.append(line)
        self._log_fn(line)

    def progress(self, frac: float, msg: str = "") -> None:
        self._progress_fn(max(0.0, min(1.0, frac)), msg)
        self.check_cancelled()

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise Cancelled("사용자가 취소함")
