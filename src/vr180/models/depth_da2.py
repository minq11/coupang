"""Depth Anything V2 (transformers). 상대 inverse depth 를 돌려준다."""

from __future__ import annotations

import numpy as np
from PIL import Image

from vr180.models.base import DepthEstimator, ModelInfo, resolve_device

_IDS = {
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
}


class DepthAnythingV2(DepthEstimator):
    def __init__(self, model_cfg: dict, variant: str = "large"):
        super().__init__(model_cfg)
        self.variant = variant
        self.repo = model_cfg.get(f"da2_{variant}_id", _IDS.get(variant, _IDS["large"]))
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.model = None
        self.processor = None

    def _load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        from vr180 import paths

        paths.ensure_hf_env()
        self.processor = AutoImageProcessor.from_pretrained(self.repo)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(self.repo, torch_dtype=dtype).to(self.device).eval()
        )

    def _unload(self) -> None:
        self.model = None
        self.processor = None

    def info(self) -> ModelInfo:
        return ModelInfo(name=f"depth_anything_v2_{self.variant}", weights=self.repo, device=self.device)

    def inverse_depth(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        self.load()
        h, w = rgb.shape[:2]
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {
            k: v.to(self.device, dtype=self.model.dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            out = self.model(**inputs)
        pred = out.predicted_depth  # (1, h', w') relative inverse depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(), size=(h, w), mode="bicubic", align_corners=False
        )[0, 0]
        return pred.detach().cpu().numpy().astype(np.float32)
