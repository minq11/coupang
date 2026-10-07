"""FLUX.1 Fill dev — GGUF Q4 (기본) 또는 NF4 양자화 + CPU 오프로드로 12GB 에 맞춘다.

라이선스: FLUX.1 [dev] 계열은 비상업(Non-Commercial) 라이선스. 상용화 시 교체 필요.
가중치: Hugging Face 에서 게이트돼 있어 HF_TOKEN 이 필요할 수 있다 (text encoder/VAE 는 base repo 에서).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from vr180.models.base import Inpainter, ModelInfo, resolve_device
from vr180.util.log import Cancelled


class FluxFillInpainter(Inpainter):
    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.repo = model_cfg.get("flux_fill_id", "black-forest-labs/FLUX.1-Fill-dev")
        self.gguf_repo = model_cfg.get("flux_fill_gguf_repo", "YarvixPA/FLUX.1-Fill-dev-GGUF")
        self.gguf_file = model_cfg.get("flux_fill_gguf_file", "flux1-fill-dev-Q4_K_S.gguf")
        self.quant = (model_cfg.get("flux_quant") or "gguf").lower()
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.pipe = None

    def _load(self) -> None:
        import torch
        from diffusers import FluxFillPipeline, FluxTransformer2DModel

        from vr180 import paths

        paths.ensure_hf_env()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        transformer = None
        if self.quant == "gguf":
            from diffusers import GGUFQuantizationConfig
            from huggingface_hub import hf_hub_download

            gguf_path = hf_hub_download(self.gguf_repo, self.gguf_file)
            transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
                config=self.repo,
                subfolder="transformer",
            )
        elif self.quant == "nf4":
            from diffusers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                self.repo, subfolder="transformer", quantization_config=bnb, torch_dtype=dtype
            )
        kw = {"torch_dtype": dtype}
        if transformer is not None:
            kw["transformer"] = transformer
        self.pipe = FluxFillPipeline.from_pretrained(self.repo, **kw)
        if self.device == "cuda":
            # T5 (약 9GB fp16) 때문에 전체 상주 불가 → 순차 CPU 오프로드
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
        w = f"{self.gguf_repo}/{self.gguf_file}" if self.quant == "gguf" else self.repo
        return ModelInfo(
            name=f"flux_fill_{self.quant}",
            weights=w,
            device=self.device,
            extra={"license": "FLUX.1-dev non-commercial"},
        )

    def inpaint(
        self, rgb, mask, prompt="", negative_prompt="", seed=0, steps=30, guidance=30.0, strength=1.0
    ):
        import torch

        self.load()
        h, w = rgb.shape[:2]
        # FLUX 는 h,w 가 16의 배수여야 함. 그렇지 않으면 diffusers 가 조용히 리사이즈해버려서
        # 원본 shape 기반 마스킹이 깨진다 → 바깥에서 16 배수로 패딩하고 추론 후 원본 크기로 crop.
        ph = (-h) % 16
        pw = (-w) % 16
        if ph or pw:
            rgb_p = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
            mask_p = np.pad(np.asarray(mask), ((0, ph), (0, pw)), mode="edge")
        else:
            rgb_p = rgb
            mask_p = np.asarray(mask)
        H, W = rgb_p.shape[:2]
        img = Image.fromarray(rgb_p)
        m = Image.fromarray((mask_p > 127).astype(np.uint8) * 255)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        # FLUX Fill 은 네거티브 프롬프트가 없고 guidance 가 SDXL 보다 훨씬 크다 (권장 30).
        g = float(guidance) if guidance >= 10 else 30.0
        _ctx = getattr(self, "ctx", None)

        def _cancel_cb(pipe, step, timestep, callback_kwargs):
            if _ctx is not None and _ctx.cancelled():
                raise Cancelled("사용자가 취소함")
            return callback_kwargs

        out = self.pipe(
            prompt=prompt or " ",
            image=img,
            mask_image=m,
            height=H,
            width=W,
            num_inference_steps=int(steps),
            guidance_scale=g,
            max_sequence_length=512,
            generator=gen,
            callback_on_step_end=_cancel_cb,
        ).images[0]
        out = np.asarray(out.convert("RGB")).copy()
        # diffusers 가 내부에서 다른 사이즈로 떨어뜨릴 수 있음 (관찰: 288 요청에 272 반환).
        # 패딩된 (H, W) 로 리사이즈 후 원본 (h, w) 로 crop → mask 와 shape 보장.
        if out.shape[:2] != (H, W):
            out = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
        out = out[:h, :w]
        keep = np.asarray(mask) <= 127
        out[keep] = rgb[keep]
        return out
