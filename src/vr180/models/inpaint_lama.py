"""LaMa (big-lama) inpaint — 큰 구멍 전용으로 설계된 가벼운 신경망. 프롬프트 없음.

푸리에 합성곱으로 넓은 영역의 구조(벽·바닥·하늘의 선)를 잘 잇는다. 1024 타일에 GPU 로 1초 안팎,
VRAM 약 2GB. 새로운 물체를 '상상' 하지는 못하므로 단순 구조 확장에 적합.

가중치는 TorchScript 로 내보낸 `big-lama.pt`(약 200MB) 하나라 **추가 패키지 없이 torch 만으로** 로드한다
(simple-lama-inpainting 패키지는 pillow<10·numpy<2 핀 때문에 쓰지 않는다). 첫 실행 때
`models/lama/big-lama.pt` 로 자동 다운로드하고, `vr180 fetch-models --lama` 로 미리 받을 수도 있다.
입력 규약: image [1,3,H,W] float 0..1, mask [1,1,H,W] 0/1 (1 = 채울 곳), H·W 는 8 의 배수(대칭 패딩).
라이선스: LaMa 코드 Apache-2.0, big-lama 가중치는 원저자(Samsung AI) 배포 조건을 따른다.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from vr180.models.base import Inpainter, ModelInfo, resolve_device

LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_FILE = "big-lama.pt"


def lama_weights_path() -> Path:
    from vr180 import paths

    return paths.models_dir() / "lama" / LAMA_FILE


def download_lama(dst: Path | None = None, log=print) -> Path:
    """big-lama.pt 를 models/lama/ 로 내려받는다 (이미 있으면 그대로)."""
    dst = dst or lama_weights_path()
    if dst.exists() and dst.stat().st_size > 100_000_000:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    log(f"[lama] {LAMA_URL} → {dst}")
    req = urllib.request.Request(LAMA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done % (20 << 20) < (1 << 20):
                log(f"[lama] {done / 1e6:.0f}/{total / 1e6:.0f} MB")
    tmp.replace(dst)
    return dst


def _pad_mod8(arr: np.ndarray) -> np.ndarray:
    """(C,H,W) 를 8 의 배수로 대칭 패딩 (LaMa 학습 규약과 동일)."""
    _, h, w = arr.shape
    ph = (-h) % 8
    pw = (-w) % 8
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="symmetric")


def run_lama(model, rgb: np.ndarray, mask_u8: np.ndarray, device: str) -> np.ndarray:
    """TorchScript big-lama 호출. rgb uint8 H×W×3, mask_u8 (255 = 채울 곳) → uint8 H×W×3 (원본 크기로 크롭)."""
    import torch

    h, w = rgb.shape[:2]
    img = _pad_mod8(np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0)
    msk = _pad_mod8((mask_u8 > 127).astype(np.float32)[None])
    img_t = torch.from_numpy(np.ascontiguousarray(img))[None].to(device)
    msk_t = (torch.from_numpy(np.ascontiguousarray(msk))[None].to(device) > 0).to(torch.float32)
    with torch.inference_mode():
        out = model(img_t, msk_t)
    out = out[0].permute(1, 2, 0).float().cpu().numpy()
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out[:h, :w]


class LaMaInpainter(Inpainter):
    MODEL_FREE = False
    NEEDS_PROMPT = False

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.model = None
        self.weights = lama_weights_path()

    def _load(self) -> None:
        try:
            import torch
        except ImportError:
            raise RuntimeError("torch 가 없습니다. `uv sync` 로 설치하세요.") from None
        if not self.weights.exists():
            log = self.ctx.log if getattr(self, "ctx", None) is not None else print
            download_lama(self.weights, log=log)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # torch.jit.load deprecation 안내 숨김
            self.model = torch.jit.load(str(self.weights), map_location=self.device)
        self.model.eval()

    def _unload(self) -> None:
        self.model = None

    def info(self) -> ModelInfo:
        return ModelInfo(name="lama_big", weights=str(self.weights), device=self.device)

    def inpaint(self, rgb, mask, prompt="", negative_prompt="", seed=0, steps=30, guidance=7.0, strength=1.0):
        self.load()
        rgb = np.asarray(rgb)
        m = (np.asarray(mask) > 127).astype(np.uint8) * 255
        if m.max() == 0:
            return rgb.copy()
        out = run_lama(self.model, rgb, m, self.device)
        result = rgb.copy()
        sel = m > 0
        result[sel] = out[sel]
        return result
