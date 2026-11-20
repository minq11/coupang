"""퀘스트 패스스루 플레이어 웹서버 (표준 라이브러리만).

  /                 player/ (index.html, app.js, vendor/three)
  /api/media        work/person/*/out/media.json 목록 (최신순)
  /media/<id>/<f>   work/person/<id>/out/<f>  (영상은 Range 요청 지원 → 퀘스트 브라우저가 앞뒤로 넘길 수 있게)

퀘스트에서 여는 법: PC 와 USB 연결 → `adb reverse tcp:8080 tcp:8080` → 퀘스트 브라우저에서 http://localhost:8080
localhost 는 보안 컨텍스트로 취급돼 HTTPS 없이 WebXR 이 열린다. 와이파이 IP 로 열려면 HTTPS 가 필요하다.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vr180 import paths

PLAYER_DIR = paths.REPO_ROOT / "player"
MIME = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".mp4": "video/mp4",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".html": "text/html; charset=utf-8",
    ".css": "text/css",
}


def media_root() -> Path:
    return paths.work_dir() / "person"


def list_media() -> list[dict]:
    root = media_root()
    if not root.exists():
        return []
    items = []
    for mj in root.glob("*/out/media.json"):
        try:
            m = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        job = mj.parent.parent.name
        m.update({"id": job, "base": f"/media/{urllib.parse.quote(job)}/", "mtime": mj.stat().st_mtime})
        items.append(m)
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


def _safe_join(base: Path, rel: str) -> Path | None:
    p = (base / rel).resolve()
    try:
        p.relative_to(base.resolve())
    except ValueError:
        return None
    return p


class Handler(SimpleHTTPRequestHandler):
    server_version = "vr180-player"

    def log_message(self, fmt, *args):  # 조용히
        pass

    def _resolve(self) -> Path | None:
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if path.startswith("/media/"):
            parts = path[len("/media/") :].split("/", 1)
            if len(parts) != 2 or not parts[0]:
                return None
            out = media_root() / parts[0] / "out"
            return _safe_join(out, parts[1])
        rel = path.lstrip("/") or "index.html"
        return _safe_join(PLAYER_DIR, rel)

    def do_GET(self):  # noqa: N802
        if urllib.parse.urlparse(self.path).path == "/api/media":
            body = json.dumps(list_media(), ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        p = self._resolve()
        if p is not None and p.is_dir():
            p = p / "index.html"
        if p is None or not p.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = p.stat().st_size
        ctype = MIME.get(p.suffix.lower()) or mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        status = HTTPStatus.OK
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            try:
                if a:
                    start = int(a)
                    end = int(b) if b else size - 1
                else:  # 끝에서 b 바이트
                    start = max(0, size - int(b))
                end = min(end, size - 1)
                if start > end:
                    raise ValueError
                status = HTTPStatus.PARTIAL_CONTENT
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)


def make_server(host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, int(port)), Handler)
    srv.daemon_threads = True
    return srv


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    srv = make_server(host, port)
    print(f"플레이어: http://localhost:{port}  (Ctrl+C 로 종료)")
    print(
        f"퀘스트: USB 연결 → `adb reverse tcp:{port} tcp:{port}` → 퀘스트 브라우저에서 http://localhost:{port}"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


_bg: dict[int, ThreadingHTTPServer] = {}


def start_background(host: str = "127.0.0.1", port: int = 8080) -> str:
    """GUI 에서 쓰는 백그라운드 서버. 이미 떠 있으면 그대로."""
    if port not in _bg:
        srv = make_server(host, port)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        _bg[port] = srv
    return f"http://localhost:{port}"
