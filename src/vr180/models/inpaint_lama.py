"""LaMa (big-lama) inpaint — 큰 구멍 전용으로 설계된 가벼운 신경망. 프롬프트 없음.

푸리에 합성곱으로 넓은 영역의 구조(벽·바닥·하늘의 선)를 잘 잇는다. 1024 타일에 GPU 로 1초 안팎,
VRAM 약 2GB. 새로운 물체를 '상상' 하지는 못하므로 단순 구조 확장에 적합.

가중치는 TorchScript 로 내보낸 `big-lama.pt`(약 200MB) 하나라 **추가 패키지 없이 torch 만으로** 로드한다
(simple-lama-inpainting 패키지는 pillow<10·numpy<2 핀 때문에 쓰지 않는다). 첫 실행 때
`models/lama/big-lama.pt` 로 자동 다운로드하고, `vr180 fetch-models --lama` 로 미리 받을 수도 있다.
입력 규약: image [1,3,H,W] float 0..1, mask [1,1,H,W] 0/1 (1 = 채울 곳), H·W 는 8 의 배수(대칭 패딩).
라이선스: LaMa 코드 Apache-2.0, big-lama 가중치는 원저자(Samsung AI) 배포 조건을 따른다.

점진(progressive) 모드 (기본 켜짐): outpaint 타일은 마스크가 타일의 60~75% 라 한 번에 돌리면 LaMa 가
무의미한 반복 질감("질감 죽")을 만든다. 그래서 먼저 stretch(늘리기+블러)로 구멍 전체를 그럴듯한 색으로
채운 placeholder 를 만들고, 원본 경계에서 `lama_band_px` 폭의 띠만 마스크로 주어 LaMa 를 돌린 뒤
그 띠를 '아는 영역' 으로 승격, 다음 띠로 넓혀 간다. LaMa 는 항상 바로 옆의 진짜/방금 만든 내용을 보며
채우므로 구조가 이어진다. 추론은 `lama_infer_size`(기본 512) 로 줄여서 하고 결과를 타일 크기로 올린다
(주변시용이라 충분하고, 띠마다 한 번씩 도는 비용을 1/4 로 줄인다).

파라미터 (models 섹션, 전부 선택): lama_progressive(true) · lama_band_px(128, 추론 해상도 기준) ·
lama_infer_size(512) · lama_max_steps(16)
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
        g = model_cfg.get
        self.device = resolve_device(g("device", "auto"))
        self.model = None
        self.weights = lama_weights_path()
        self.progressive = bool(g("lama_progressive", True))
        self.band_px = int(g("lama_band_px", 128))
        self.infer_size = int(g("lama_infer_size", 512))
        self.max_steps = int(g("lama_max_steps", 16))

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
        return ModelInfo(
            name="lama_big",
            weights=str(self.weights),
            device=self.device,
            extra={"progressive": self.progressive, "band_px": self.band_px, "infer_size": self.infer_size},
        )

    def inpaint(self, rgb, mask, prompt="", negative_prompt="", seed=0, steps=30, guidance=7.0, strength=1.0):
        self.load()
        rgb = np.asarray(rgb)
        m = (np.asarray(mask) > 127).astype(np.uint8) * 255
        if m.max() == 0:
            return rgb.copy()
        if not self.progressive:
            out = run_lama(self.model, rgb, m, self.device)
        else:
            out = self._progressive(rgb, m)
        result = rgb.copy()
        sel = m > 0
        result[sel] = out[sel]
        return result

    def _progressive(self, rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
        import cv2

        from vr180.models.inpaint_stretch import StretchBlurInpainter

        h, w = rgb.shape[:2]
        # 1) placeholder: 구멍 전체를 stretch 로 그럴듯한 색으로 채운다 (LaMa 가 '아는 영역' 으로 볼 배경)
        cur = StretchBlurInpainter(self.cfg).inpaint(rgb, m)
        # 2) 추론 해상도로 축소
        scale = min(1.0, self.infer_size / float(max(h, w)))
        if scale < 1.0:
            sw, sh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
            cur_s = cv2.resize(cur, (sw, sh), interpolation=cv2.INTER_AREA)
            hole_s = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST) > 127
        else:
            cur_s, hole_s = cur.copy(), m > 127
        dist = cv2.distanceTransform(hole_s.astype(np.uint8), cv2.DIST_L2, 5)
        remaining = hole_s.copy()
        step = 0
        while remaining.any() and step < self.max_steps:
            band = remaining & (dist <= float(self.band_px) * (step + 1))
            if not band.any():
                break
            out_s = run_lama(self.model, cur_s, band.astype(np.uint8) * 255, self.device)
            cur_s[band] = out_s[band]
            remaining &= ~band
            step += 1
            if getattr(self, "ctx", None) is not None:
                self.ctx.check_cancelled()
        if remaining.any():  # max_steps 에 걸린 나머지는 한 번에
            out_s = run_lama(self.model, cur_s, remaining.astype(np.uint8) * 255, self.device)
            cur_s[remaining] = out_s[remaining]
        # 3) 타일 크기로 복원
        if scale < 1.0:
            return cv2.resize(cur_s, (w, h), interpolation=cv2.INTER_CUBIC)
        return cur_s
