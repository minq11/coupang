"""이미지/배열 입출력. Windows 의 한글 경로에서도 동작하도록 cv2.imread 대신 imdecode 를 쓴다.

색 순서는 패키지 전체에서 RGB(A) 로 통일한다 (OpenCV BGR 은 여기서만 변환).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_rgb(path: Path | str, keep_alpha: bool = False) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {path}")
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if keep_alpha:
            img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img if keep_alpha else img[..., :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if keep_alpha:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return img


def read_rgba(path: Path | str) -> np.ndarray:
    return read_rgb(path, keep_alpha=True)


def read_mask(path: Path | str) -> np.ndarray:
    """단일 채널 uint8 (255 = 유효)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"마스크를 읽을 수 없음: {path}")
    return img


def write_image(path: Path | str, img: np.ndarray, jpg_quality: int = 95) -> Path:
    """RGB / RGBA / 단일채널 uint8 을 확장자에 맞게 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ascontiguousarray(img)
    if img.ndim == 3 and img.shape[2] == 4:
        enc = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        if path.suffix.lower() in (".jpg", ".jpeg"):
            enc = enc[..., :3]
    elif img.ndim == 3 and img.shape[2] == 3:
        enc = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        enc = img
    params = []
    if path.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
    elif path.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(path.suffix, enc, params)
    if not ok:
        raise RuntimeError(f"인코딩 실패: {path}")
    buf.tofile(str(path))
    return path


def write_npy(path: Path | str, arr: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.asarray(arr, dtype=np.float32))
    return path


def read_npy(path: Path | str) -> np.ndarray:
    return np.load(str(path))


def write_json(path: Path | str, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"JSON 직렬화 불가: {type(o)}")


def colormap_preview(
    arr: np.ndarray, valid: np.ndarray | None = None, vmin: float | None = None, vmax: float | None = None
) -> tuple[np.ndarray, dict]:
    """float 맵 → turbo 컬러맵 RGB 미리보기. 범위를 같이 돌려준다 (JSON 기록용)."""
    arr = np.asarray(arr, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(arr)
    vals = arr[valid]
    if vals.size == 0:
        vals = np.array([0.0, 1.0], dtype=np.float32)
    lo = float(np.percentile(vals, 1)) if vmin is None else float(vmin)
    hi = float(np.percentile(vals, 99)) if vmax is None else float(vmax)
    if hi - lo < 1e-12:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb, {"vmin": lo, "vmax": hi, "colormap": "turbo"}


def checkerboard_background(rgba: np.ndarray, cell: int = 32) -> np.ndarray:
    """RGBA 의 투명 영역을 체크무늬로 채운 RGB (GUI 표시용)."""
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // cell) + (xx // cell)) % 2).astype(np.uint8)
    bg = np.where(checker[..., None] == 0, 90, 140).astype(np.uint8)
    bg = np.repeat(bg, 3, axis=-1)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3].astype(np.float32) * a + bg.astype(np.float32) * (1 - a)).astype(np.uint8)


def flatten_rgba(rgba: np.ndarray, background: str = "black") -> np.ndarray:
    color = {"black": 0, "gray": 128, "white": 255}.get(background, 0)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3].astype(np.float32) * a + color * (1 - a)).astype(np.uint8)


def downscale_for_preview(img: np.ndarray, max_side: int = 2048) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    f = max_side / s
    return cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
