"""플레이어 웹서버: 목록, 영상 구간 요청, 경로 탈출 차단, JS 형식."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from vr180 import player_server


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("VR180_WORK_DIR", str(tmp_path))
    out = tmp_path / "person" / "job1" / "out"
    out.mkdir(parents=True)
    (out / "media.json").write_text(
        json.dumps({"layout": "rgba_sbs_v1", "type": "video", "file": "a.mp4", "people": 2})
    )
    (out / "a.mp4").write_bytes(bytes(range(256)) * 4)
    (tmp_path / "secret.txt").write_text("x")
    srv = player_server.make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=5)


def test_media_list_and_range(server):
    items = json.loads(_get(server + "/api/media").read())
    assert items[0]["id"] == "job1" and items[0]["base"] == "/media/job1/"
    r = _get(server + "/media/job1/a.mp4", {"Range": "bytes=10-19"})
    assert r.status == 206 and r.read() == bytes(range(10, 20))
    assert r.headers["Content-Range"] == "bytes 10-19/1024"
    full = _get(server + "/media/job1/a.mp4")
    assert full.status == 200 and len(full.read()) == 1024 and full.headers["Accept-Ranges"] == "bytes"


def test_player_files_and_no_traversal(server):
    r = _get(server + "/")
    assert r.status == 200 and b"importmap" in r.read()
    js = _get(server + "/vendor/three/three.module.min.js")
    assert js.headers["Content-Type"] == "text/javascript"
    for bad in (
        "/media/job1/..%2F..%2F..%2Fsecret.txt",
        "/..%2Fsrc%2Fvr180%2Fpaths.py",
        "/media/../../secret.txt",
    ):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(server + bad)
        assert e.value.code == 404
