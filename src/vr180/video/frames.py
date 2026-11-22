"""ffmpeg 디코드/인코드 파이프. subprocess 로만 호출하고 PyAV 는 안 쓴다 (추가 의존성 피함).

decoder: ffmpeg stdout 에서 rawvideo rgb24 로 프레임을 읽는다.
encoder: ffmpeg stdin 으로 rawvideo rgb24 를 보낸다 (오디오는 입력에서 -c:a copy).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path


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
    """ffprobe 로 비디오 스트림 정보 추출. 반환: {width, height, fps, nb_frames, duration}.

    mkv 는 스트림 duration/nb_frames 가 N/A 인 경우가 많아 컨테이너(format) duration 으로 보완한다."""
    _ensure_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
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

    def _f(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):  # "N/A" 등
            return 0.0

    dur = _f(s.get("duration")) or _f(data.get("format", {}).get("duration"))
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
        # 호출자가 중간에 그만 읽으면 (샷 안정성 확인은 300 프레임만 본다) ffmpeg 가 아직 쓰는 중이다.
        # stdout 을 먼저 닫으면 ffmpeg 가 'Broken pipe' 오류를 콘솔에 쏟으므로 먼저 종료시킨다.
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.wait()


class EncoderPipe:
    """stdin 에 rgb24 바이트를 보내 H.265 영상으로 인코드. 오디오는 원본에서 복사.

    ffmpeg 의 stderr 는 별도 스레드가 계속 비운다. 비우지 않으면 ffmpeg 가 배너·스트림 정보(≈4KB 이상)를 쓰다
    Windows 파이프 버퍼(4KB)가 차서 멈추고, 우리는 stdin.write 에서, 디코더는 stdout 에서 서로를 기다리는 교착이 난다
    (출력 mp4 가 44 바이트 헤더만 남는 증상)."""

    def __init__(
        self,
        src: Path,
        out: Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx265",
        crf: int = 20,
        pix_fmt: str = "rgb24",
    ):
        _ensure_ffmpeg()
        self.pix_fmt = pix_fmt  # 파이프로 들어오는 raw 프레임 형식: rgb24 (W·H·3) 또는 yuv420p (W·H·1.5, I420 평면)
        self.src = Path(src)
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = codec
        self.crf = int(crf)
        self.proc: subprocess.Popen | None = None
        self._stderr_tail: deque[bytes] = deque(maxlen=200)  # 마지막 200 줄만 보관 (오류 보고용)
        self._stderr_thread: threading.Thread | None = None

    def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in iter(self.proc.stderr.readline, b""):
            self._stderr_tail.append(line)

    def stderr_text(self) -> str:
        return b"".join(self._stderr_tail).decode(errors="replace")

    def _quality_args(self) -> list[str]:
        """x264/x265 는 crf. NVENC(hevc_nvenc 등)는 crf 가 없어 같은 숫자를 cq(고정 품질)로 쓴다.
        NVENC 는 CPU 를 거의 안 써서 인물 AR 처럼 프레임 만들기가 CPU 를 다 쓰는 경로에서 빠르다."""
        if "nvenc" in self.codec:
            return ["-rc", "vbr", "-cq", str(self.crf), "-b:v", "0", "-preset", "p5"]
        return ["-crf", str(self.crf), "-preset", "medium"]

    def __enter__(self):
        # 입력 1: rawvideo 파이프 (영상). 입력 2: 원본 파일 (오디오만 복사).
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.pix_fmt,
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
            *self._quality_args(),
            "-c:a",
            "copy",
            "-shortest",
        ]
        if "265" in self.codec or "hevc" in self.codec:
            # hev1 태그는 일부 VR 플레이어가 거부한다. hvc1 (파라미터 셋을 hvcC 에) 로 기록.
            cmd += ["-tag:v", "hvc1"]
        cmd.append(str(self.out))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        return self

    def write(self, buf: bytes) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.write(buf)
        except (BrokenPipeError, OSError) as e:
            # ffmpeg 가 먼저 죽었다 (코덱/크기 오류 등). 이유를 stderr 에서 가져와 바로 알린다.
            rc = self.proc.wait()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=5)
            raise RuntimeError(f"ffmpeg 인코더가 중단됨 (rc={rc}): {self.stderr_text()[-500:]}") from e

    def __exit__(self, et, ev, tb):
        if self.proc is None:
            return False
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        rc = self.proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        if rc != 0 and et is None:  # 이미 예외 전파 중이면 덮어쓰지 않는다
            raise RuntimeError(f"ffmpeg 인코딩 실패 (rc={rc}): {self.stderr_text()[-500:]}")
        return False

def extract_frames_jpg(
    input_path: Path, out_dir: Path, width: int, height: int, quality: int = 2
) -> int:
    """영상 전체를 out_dir/00000.jpg … 로 뽑는다 (SAM 2 영상 입력 형식). 지정 크기로 축소. 반환: 프레임 수."""
    _ensure_ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(input_path),
        "-vf", f"scale={int(width)}:{int(height)}:flags=lanczos",
        "-q:v", str(int(quality)), "-start_number", "0",
        str(out_dir / "%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.jpg")))
