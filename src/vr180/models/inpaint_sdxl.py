"""SDXL inpainting (fp16, 약 7GB VRAM). FLUX 가 안 들어가거나 느릴 때의 대안."""

from __future__ import annotations

import numpy as np
from PIL import Image

from vr180.models.base import Inpainter, ModelInfo, resolve_device


class SDXLInpainter(Inpainter):
    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.repo = model_cfg.get("sdxl_inpaint_id", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.pipe = None

    def _load(self) -> None:
        import torch
        from diffusers import AutoPipelineForInpainting

        from vr180 import paths

        paths.ensure_hf_env()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            self.repo,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            try:
                self.pipe.enable_vae_tiling()
            except Exception:
                pass
        else:
            self.pipe.to("cpu")

    def _unload(self) -> None:
        self.pipe = None

    def info(self) -> ModelInfo:
        return ModelInfo(name="sdxl_inpaint", weights=self.repo, device=self.device)

    def inpaint(self, rgb, mask, prompt="", negative_prompt="", seed=0, steps=30, guidance=7.0, strength=1.0):
        import torch

        self.load()
        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb)
        m = Image.fromarray((np.asarray(mask) > 127).astype(np.uint8) * 255)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt or " ",
            negative_prompt=negative_prompt or None,
            image=img,
            mask_image=m,
            height=h,
            width=w,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            strength=float(strength),
            generator=gen,
        ).images[0]
        out = np.asarray(out.convert("RGB"))
        # 마스크 밖은 원본 유지 (VAE 왕복으로 인한 미세 변화 제거)
        keep = np.asarray(mask) <= 127
        out = out.copy()
        out[keep] = rgb[keep]
        return out
