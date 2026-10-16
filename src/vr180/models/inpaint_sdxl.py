"""SDXL inpainting (fp16, 약 7GB VRAM). FLUX 가 안 들어가거나 느릴 때의 대안.

variant:
  base      — 기본 30 스텝, guidance 7. 10~20초/타일.
  lcm       — LCM-LoRA (latent-consistency/lcm-lora-sdxl) + LCMScheduler. 4~8 스텝, guidance 1~2. 3~5초/타일.
  lightning — SDXL-Lightning 4-step LoRA (ByteDance) + Euler trailing. 4 스텝, guidance 0. 2~4초/타일.
가속 변형은 같은 SDXL inpainting 가중치 위에 LoRA 만 얹으므로 추가 VRAM 이 거의 없다.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from vr180.models.base import Inpainter, ModelInfo, resolve_device
from vr180.util.log import Cancelled


class SDXLInpainter(Inpainter):
    FAST_DEFAULTS = {"lcm": (8, 1.5), "lightning": (4, 0.0)}  # variant → (steps, guidance)

    def __init__(self, model_cfg: dict, variant: str = "base"):
        super().__init__(model_cfg)
        self.repo = model_cfg.get("sdxl_inpaint_id", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
        self.variant = (variant or "base").lower()
        if self.variant not in ("base", "lcm", "lightning"):
            raise ValueError(f"알 수 없는 SDXL variant: {variant}")
        self.lcm_lora = model_cfg.get("sdxl_lcm_lora_id", "latent-consistency/lcm-lora-sdxl")
        self.lightning_repo = model_cfg.get("sdxl_lightning_repo", "ByteDance/SDXL-Lightning")
        self.lightning_file = model_cfg.get("sdxl_lightning_file", "sdxl_lightning_4step_lora.safetensors")
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
        # 가속 LoRA 는 오프로드 전에 올리고 fuse 해 둔다 (추론 중 LoRA 연산 비용 제거)
        if self.variant == "lcm":
            from diffusers import LCMScheduler

            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.load_lora_weights(self.lcm_lora)
            self.pipe.fuse_lora()
        elif self.variant == "lightning":
            from diffusers import EulerDiscreteScheduler

            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="trailing"
            )
            self.pipe.load_lora_weights(self.lightning_repo, weight_name=self.lightning_file)
            self.pipe.fuse_lora()
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
        name = "sdxl_inpaint" if self.variant == "base" else f"sdxl_inpaint_{self.variant}"
        extra = {}
        if self.variant == "lcm":
            extra["lora"] = self.lcm_lora
        elif self.variant == "lightning":
            extra["lora"] = f"{self.lightning_repo}/{self.lightning_file}"
        return ModelInfo(name=name, weights=self.repo, device=self.device, extra=extra)

    def _effective_steps_guidance(self, steps: int, guidance: float) -> tuple[int, float]:
        """가속 변형은 스텝·guidance 범위가 다르다. 호출자가 base 기본값(30/7)을 넘기면 변형 기본값으로 바꾼다."""
        if self.variant == "base":
            return int(steps), float(guidance)
        d_steps, d_guid = self.FAST_DEFAULTS[self.variant]
        steps = d_steps if int(steps) > 12 else max(1, int(steps))
        if self.variant == "lcm":
            guidance = d_guid if guidance > 2.5 else float(np.clip(guidance, 1.0, 2.0))
        else:  # lightning: CFG 없음
            guidance = 0.0
        return steps, guidance

    def inpaint(self, rgb, mask, prompt="", negative_prompt="", seed=0, steps=30, guidance=7.0, strength=1.0):
        import torch

        self.load()
        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb)
        m = Image.fromarray((np.asarray(mask) > 127).astype(np.uint8) * 255)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        steps, guidance = self._effective_steps_guidance(steps, guidance)
        _ctx = getattr(self, "ctx", None)

        def _cancel_cb(pipe, step, timestep, callback_kwargs):
            if _ctx is not None and _ctx.cancelled():
                raise Cancelled("사용자가 취소함")
            return callback_kwargs

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
            callback_on_step_end=_cancel_cb,
        ).images[0]
        out = np.asarray(out.convert("RGB"))
        # 마스크 밖은 원본 유지 (VAE 왕복으로 인한 미세 변화 제거)
        keep = np.asarray(mask) <= 127
        out = out.copy()
        out[keep] = rgb[keep]
        return out
