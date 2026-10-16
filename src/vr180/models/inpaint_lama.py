"""LaMa (big-lama) inpaint — 큰 구멍 전용으로 설계된 가벼운 신경망. 프롬프트 없음.

푸리에 합성곱으로 넓은 영역의 구조(벽·바닥·하늘의 선)를 잘 잇는다. 1024 타일에 GPU 로 1초 안팎,
VRAM 약 2GB. 새로운 물체를 '상상' 하지는 못하므로 단순 구조 확장에 적합.

설치: `uv sync --extra lama`  (simple-lama-inpainting). 가중치(big-lama.pt, ~200MB)는 첫 실행 때
models/torch_hub 아래로 내려받는다. `models/lama/big-lama.pt` 가 있으면 그걸 쓴다.
라이선스: LaMa 코드 Apache-2.0, big-lama 가중치는 원저자(Samsung AI) 배포 조건을 따른다.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from vr180.models.base import Inpainter, ModelInfo, resolve_device


class LaMaInpainter(Inpainter):
    MODEL_FREE = False
    NEEDS_PROMPT = False

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.model = None

    def _load(self) -> None:
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as e:  # noqa: F841
            raise RuntimeError(
                "LaMa 가 설치되지 않았습니다. `uv sync --extra lama` 로 설치하거나 inpainter 를 바꾸세요."
            ) from None
        import torch

        from vr180 import paths

        torch.hub.set_dir(str(paths.models_dir() / "torch_hub"))
        custom = paths.models_dir() / "lama" / "big-lama.pt"
        if custom.exists():
            os.environ.setdefault("LAMA_MODEL", str(custom))
        self.model = SimpleLama(device=torch.device(self.device))

    def _unload(self) -> None:
        self.model = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="lama_big",
            weights=os.environ.get("LAMA_MODEL", "big-lama.pt (torch hub)"),
            device=self.device,
        )

    def inpaint(self, rgb, mask, prompt="", negative_prompt="", seed=0, steps=30, guidance=7.0, strength=1.0):
        self.load()
        rgb = np.asarray(rgb)
        m = (np.asarray(mask) > 127).astype(np.uint8) * 255
        if m.max() == 0:
            return rgb.copy()
        h, w = rgb.shape[:2]
        out = self.model(Image.fromarray(rgb), Image.fromarray(m))
        out = np.asarray(out.convert("RGB"))
        # simple-lama 는 8 배수로 패딩한다. 크기가 다르면 원본 크기로 맞춘다.
        if out.shape[0] >= h and out.shape[1] >= w:
            out = out[:h, :w]
        elif out.shape[:2] != (h, w):
            import cv2

            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        result = rgb.copy()
        sel = m > 0
        result[sel] = out[sel]
        return result
