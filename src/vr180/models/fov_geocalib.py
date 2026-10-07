"""GeoCalib 단일 이미지 카메라 추정 → 수평 FOV. 선택 의존성 (`uv sync --extra geocalib`)."""

from __future__ import annotations

import math

import numpy as np

from vr180.models.base import FovEstimator, ModelInfo, resolve_device


class GeoCalibFov(FovEstimator):
    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.model = None

    def _load(self) -> None:
        from geocalib import GeoCalib  # type: ignore

        self.model = GeoCalib().to(self.device)

    def _unload(self) -> None:
        self.model = None

    def info(self) -> ModelInfo:
        return ModelInfo(name="geocalib", weights="cvg/GeoCalib pinhole", device=self.device)

    def estimate_hfov(self, rgb: np.ndarray) -> tuple[float | None, float]:
        import torch

        self.load()
        img = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(self.device) / 255.0
        try:
            with torch.inference_mode():
                res = self.model.calibrate(img)
        except Exception:
            return None, 0.0
        cam = res["camera"]
        w = float(rgb.shape[1])
        f = float(cam.f[0, 0].detach().cpu())  # 픽셀 단위 fx
        hfov = math.degrees(2.0 * math.atan(w / (2.0 * f)))
        # focal 불확실도 → 0..1 신뢰도 (대략)
        unc = res.get("focal_uncertainty")
        conf = 0.5
        if unc is not None:
            u = float(torch.as_tensor(unc).flatten()[0].detach().cpu())
            conf = float(max(0.0, min(1.0, 1.0 - u / max(f, 1.0) * 10.0)))
        return hfov, conf
