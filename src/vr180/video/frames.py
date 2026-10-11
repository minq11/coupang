"""ffmpeg 디코드/인코드 파이프. subprocess 로만 호출하고 PyAV 는 안 쓴다 (추가 의존성 피함).

decoder: ffmpeg stdout 에서 rawvideo rgb24 로 프레임을 읽는다.
encoder: ffmpeg stdin 으로 rawvideo rgb24 를 보낸다 (오디오는 입력에서 -c:a copy).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterator


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class FFmpegMissingError(RuntimeError):
    """ffmpeg/ffprobe 를 찾을 수 없을 때."""


def _ensure_ffmpeg() -> None:
    if not have_ffmpeg():
        raise FFmpegMissingError(
            "ffmpeg 와 ffprobe 가 PATH 에 있어야 합니다. Windows: `winget install ffmpeg` 또는 공식 zip 다운로드."
        )


def probe(path: Path) -> dict:
    """ffprobe 로 비디오 스트림 정보 추출. 반환: {width, height, fps, nb_frames, duration}."""
    _ensure_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(res.stdout)
    s = data["streams"][0]
    num, den = (s["r_frame_rate"].split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    try:
        nb = int(s.get("nb_frames") or 0)
    except Exception:
        nb = 0
    dur = float(s.get("duration") or 0.0)
    if nb == 0 and dur > 0:
        nb = int(round(dur * fps))
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "nb_frames": nb, "duration": dur}


def extract_frame_png(input_path: Path, time_sec: float, out_png: Path) -> Path:
    """특정 시각의 1프레임만 PNG 로 뽑는다 (기준 프레임용)."""
    _ensure_ffmpeg()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        str(input_path),
        "-frames:v",
        "1",
        "-f",
        "image2",
        str(out_png),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_png


def frame_iter(input_path: Path, width: int, height: int) -> Iterator[bytes]:
    """rawvideo rgb24 바이트를 프레임 단위로 yield. 호출자가 numpy 로 바꿔 쓴다."""
    _ensure_ffmpeg()
    frame_size = width * height * 3
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_size * 2)
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            yield buf
    finally:
        proc.stdout.close()
        proc.wait()


class EncoderPipe:
    """stdin 에 rgb24 바이트를 보내 H.265 영상으로 인코드. 오디오는 원본에서 복사."""

    def __init__(
        self,
        src: Path,
        out: Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx265",
        crf: int = 20,
    ):
        _ensure_ffmpeg()
        self.src = Path(src)
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = codec
        self.crf = int(crf)
        self.proc: subprocess.Popen | None = None

    def __enter__(self):
        # 입력 1: rawvideo 파이프 (영상). 입력 2: 원본 파일 (오디오만 복사).
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps}",
            "-i",
            "-",
            "-i",
            str(self.src),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            self.codec,
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(self.crf),
            "-preset",
            "medium",
            "-c:a",
            "copy",
            "-shortest",
            str(self.out),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def write(self, buf: bytes) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(buf)

    def __exit__(self, et, ev, tb):
        if self.proc is not None and self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            rc = self.proc.wait()
            if rc != 0:
                err = (self.proc.stderr.read() if self.proc.stderr else b"").decode(errors="replace")
                raise RuntimeError(f"ffmpeg 인코딩 실패 (rc={rc}): {err[-500:]}")
