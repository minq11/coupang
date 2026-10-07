"""파일 / 파라미터 해시 — Stage 캐시 키."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_hash(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=16)
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(f.relative_to(p).as_posix().encode())
                h.update(file_hash(f).encode())
        return h.hexdigest()
    with open(p, "rb") as fp:
        while True:
            b = fp.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def files_hash(paths: list[Path]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in paths:
        p = Path(p)
        h.update(p.name.encode())
        h.update(file_hash(p).encode() if p.exists() else b"<missing>")
    return h.hexdigest()


def params_hash(params: dict[str, Any]) -> str:
    s = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()
