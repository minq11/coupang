"""영상 변환 end-to-end — ffmpeg 가 있을 때만 실행. 합성 SBS 프레임 10장으로."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from vr180.sample import make_sbs
from vr180.util import imageio
from vr180.util.log import RunContext
from vr180.video import frames as _frames


def _ctx():
    return RunContext(log_fn=lambda s: None)


def _ffmpeg_available() -> bool:
    try:
        return _frames.have_ffmpeg()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH")


def _make_tiny_sbs_video(dst: Path, n_frames: int = 10, w: int = 240, h: int = 136, fps: int = 10) -> Path:
    """합성 SBS 10 프레임을 PNG 로 만들고 ffmpeg 로 묶는다 (fps=10, 코덱=mpeg4)."""
    tmp = dst.parent / "frames"
    tmp.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        img = make_sbs(w // 2, h, 80.0, 0.065, seed=i)
        imageio.write_image(tmp / f"f{i:04d}.png", img)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp / "f%04d.png"),
            "-c:v",
            "mpeg4",
            "-qscale:v",
            "3",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    return dst


def test_video_sbs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("VR180_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("VR180_MODELS_DIR", str(tmp_path / "models"))
    src = _make_tiny_sbs_video(tmp_path / "src.mp4", n_frames=10, w=240, h=136, fps=10)
    out = tmp_path / "out.mp4"
    from vr180.video.run import run_video

    override = {
        "stage_02_reproj": {"size": 256, "extent_h_deg": 180, "extent_v_deg": 180, "extent_fade_deg": 0},
        "stage_03_outpaint": {"tile_res": 128, "feather_px": 16, "inpainter": "opencv"},
        "stage_04_stereo": {"matcher": "sgbm", "max_disparity": 64},
        "stage_05_depth": {"estimator": "none"},
        "stage_06_right": {"inpainter": "opencv"},
    }
    res = run_video(
        src,
        out,
        ref_frame=0,
        input_mode="sbs",
        codec="mpeg4",
        crf=10,
        check_motion=False,
        detect_cuts=False,
        params_override=override,
        ctx=_ctx(),
    )
    assert out.exists()
    assert res["frames"] == 10
    info_out = _frames.probe(out)
    # 눈당 256 → 전체 512 × 256
    assert info_out["width"] == 512 and info_out["height"] == 256
    assert info_out["nb_frames"] == 10 or abs((info_out.get("duration") or 1.0) - 1.0) < 0.3
