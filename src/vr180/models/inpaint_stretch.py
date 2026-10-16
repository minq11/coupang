"""늘리기 + 거리 비례 블러 (stretch-blur). 모델 없이 밀리초 단위로 주변부를 채운다.

원리: 원본 가장자리 픽셀을 바깥으로 쭉 늘린 뒤(최근접 유효 픽셀 복사), 경계에서 멀어질수록
강하게 블러한다. 구조는 못 만들지만 색·밝기가 끊기지 않아 주변시(周邊視) 용으로는 충분하고,
TV 뒤 앰비언트 조명 같은 느낌이 난다. 범위 제한 + 페이드와 궁합이 가장 좋다.

파라미터 (models 섹션, 전부 선택):
  stretch_sigma_min_frac  경계 바로 바깥의 블러 σ (타일 긴 변 대비 비율). 기본 0.01
  stretch_sigma_max_frac  가장 먼 곳의 블러 σ. 기본 0.08
  stretch_reach_frac      σ 가 max 에 도달하는 거리 (긴 변 대비). 기본 0.5
  stretch_darken          가장 먼 곳에서 어둡게 할 비율 0..1. 기본 0.15
  stretch_seam_frac       경계에서 이 거리까지는 늘린 질감을 그대로 두고 서서히 블러로 녹인다 (이음새 완화). 기본 0.03
  stretch_levels          블러 단계 수. 기본 5
"""

from __future__ import annotations

import cv2
import numpy as np

from vr180.models.base import Inpainter, ModelInfo
from vr180.util.fill import nearest_fill_rgb


class StretchBlurInpainter(Inpainter):
    MODEL_FREE = True
    NEEDS_PROMPT = False

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        g = model_cfg.get
        self.sigma_min_frac = float(g("stretch_sigma_min_frac", 0.01))
        self.sigma_max_frac = float(g("stretch_sigma_max_frac", 0.08))
        self.reach_frac = float(g("stretch_reach_frac", 0.5))
        self.darken = float(np.clip(g("stretch_darken", 0.15), 0.0, 1.0))
        self.seam_frac = float(g("stretch_seam_frac", 0.03))
        self.levels = max(2, int(g("stretch_levels", 5)))

    def _load(self) -> None:
        pass

    def _unload(self) -> None:
        pass

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="stretch_blur",
            device="cpu",
            extra={
                "sigma_min_frac": self.sigma_min_frac,
                "sigma_max_frac": self.sigma_max_frac,
                "reach_frac": self.reach_frac,
                "darken": self.darken,
                "seam_frac": self.seam_frac,
            },
        )

    def inpaint(self, rgb, mask, prompt="", negative_prompt="", seed=0, steps=30, guidance=7.0, strength=1.0):
        rgb = np.asarray(rgb)
        hole = np.asarray(mask) > 127
        if not hole.any():
            return rgb.copy()
        valid = ~hole
        if not valid.any():
            return rgb.copy()
        h, w = rgb.shape[:2]
        long_side = float(max(h, w))

        # 1) 가장자리 픽셀을 바깥으로 늘린다 (1차원 줄무늬가 생기지만 블러가 숨긴다)
        stretched = nearest_fill_rgb(rgb, valid).astype(np.float32)

        # 2) 경계에서의 거리 → 블러 세기 0..1
        dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
        reach = max(1.0, self.reach_frac * long_side)
        t = np.clip(dist / reach, 0.0, 1.0).astype(np.float32)

        # 3) 블러 피라미드: σ 를 로그 간격으로 levels 개 만들고 t 로 선형 보간 (텐트 가중치)
        s_min = max(0.5, self.sigma_min_frac * long_side)
        s_max = max(s_min, self.sigma_max_frac * long_side)
        sigmas = np.exp(np.linspace(np.log(s_min), np.log(s_max), self.levels))
        out = np.zeros_like(stretched)
        wsum = np.zeros((h, w), np.float32)
        pos = t * (self.levels - 1)
        for k, sigma in enumerate(sigmas):
            wk = np.clip(1.0 - np.abs(pos - k), 0.0, 1.0)
            if not wk.any():
                continue
            blurred = _gaussian(stretched, float(sigma))
            out += blurred * wk[..., None]
            wsum += wk
        out /= np.maximum(wsum, 1e-6)[..., None]

        # 3b) 이음새 완화: 경계 바로 바깥은 늘린 질감을 그대로 두고 seam 거리까지 블러로 서서히 녹인다
        seam = self.seam_frac * long_side
        if seam > 0.5:
            ws = np.clip(dist / seam, 0.0, 1.0).astype(np.float32)[..., None]
            out = stretched * (1.0 - ws) + out * ws

        # 4) 멀수록 살짝 어둡게 (주변부가 원본보다 튀지 않게)
        if self.darken > 0:
            out *= (1.0 - self.darken * t)[..., None]

        result = rgb.copy()
        result[hole] = np.clip(out[hole], 0, 255).astype(np.uint8)
        return result


def _gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """큰 σ 는 축소해서 블러한 뒤 되돌려 시간을 아낀다 (결과는 어차피 저주파)."""
    h, w = img.shape[:2]
    if sigma > 12:
        f = 12.0 / sigma
        small = cv2.resize(img, (max(2, int(w * f)), max(2, int(h * f))), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 12.0, borderType=cv2.BORDER_REPLICATE)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
