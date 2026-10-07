"""OpenCV Telea inpaint. 모델 없이 파이프라인 뼈대를 끝까지 돌리기 위한 구현 + Stage 6 작은 구멍용."""

from __future__ import annotations

import cv2
import numpy as np

from vr180.models.base import Inpainter, ModelInfo


class OpenCVInpainter(Inpainter):
    def __init__(self, model_cfg: dict, radius: int = 5):
        super().__init__(model_cfg)
        self.radius = radius

    def _load(self) -> None:
        pass

    def _unload(self) -> None:
        pass

    def info(self) -> ModelInfo:
        return ModelInfo(name="opencv_telea", weights="", device="cpu", extra={"radius": self.radius})

    def inpaint(self, rgb, mask, prompt="", negative_prompt="", seed=0, steps=30, guidance=7.0, strength=1.0):
        mask_u8 = (np.asarray(mask) > 127).astype(np.uint8) * 255
        if mask_u8.max() == 0:
            return rgb.copy()
        # 큰 영역은 Telea 가 느리고 번지므로 1/4 로 줄여 채운 뒤 올린다 (뼈대용 품질)
        h, w = rgb.shape[:2]
        if mask_u8.mean() > 40 and max(h, w) > 512:
            small = cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            msmall = cv2.resize(mask_u8, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
            filled = cv2.inpaint(small, msmall, self.radius, cv2.INPAINT_TELEA)
            filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
            out = rgb.copy()
            out[mask_u8 > 0] = filled[mask_u8 > 0]
            return out
        return cv2.inpaint(rgb, mask_u8, self.radius, cv2.INPAINT_TELEA)
