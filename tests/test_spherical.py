"""mp4 VR180 메타데이터(st3d/sv3d) 삽입 테스트. ffmpeg/ffprobe 가 있어야 한다."""

import json
import shutil
import subprocess

import pytest

from vr180.video.spherical import inject_vr180

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg 없음"
)


def _make(tmp_path, name, *extra):
    out = tmp_path / name
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=128x64:rate=10",
            "-t",
            "0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            *extra,
            str(out),
        ],
        check=True,
    )
    return out


def _side_data(path):
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream_side_data_list",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return {d["side_data_type"]: d for d in json.loads(p.stdout)["streams"][0].get("side_data_list", [])}


@pytest.mark.parametrize("faststart", [False, True])
def test_inject_vr180_recognised_by_ffprobe(tmp_path, faststart):
    extra = ["-movflags", "+faststart"] if faststart else []
    f = _make(tmp_path, "t.mp4", *extra)
    assert _side_data(f) == {}
    r = inject_vr180(f)
    assert r["sample_entry"] == "avc1" and r["moov_before_mdat"] is faststart
    if faststart:
        assert r["chunk_offsets_shifted"] > 0
    sd = _side_data(f)
    assert sd["Stereo 3D"]["type"] == "side by side"
    assert sd["Spherical Mapping"]["projection"] == "tiled equirectangular"
    # 전체 디코드가 되어야 (오프셋 보정 검증)
    d = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(f), "-f", "null", "-"], capture_output=True, text=True
    )
    assert d.returncode == 0 and d.stderr == ""
    # 다시 넣으면 기존 것을 빼고 교체 (중복 없음)
    r2 = inject_vr180(f)
    assert r2["removed"] == r["inserted"]
    assert _side_data(f)["Stereo 3D"]["type"] == "side by side"
