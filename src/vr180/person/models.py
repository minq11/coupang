"""인물 AR 모델 래퍼. 무거운 import 는 load() 안에서만 한다 (모델 없이도 나머지 파이프라인과 테스트가 돈다).

- PersonDetector : RGB → 사람 마스크 목록       (RFDETRSegDetector)
- Matter         : RGB → 0..1 알파              (BiRefNetMatter)
- load_sam2_video_predictor : SAM 2.1 영상 추적기 (third_party/sam2 소스 + models/sam2 가중치)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vr180 import paths
from vr180.models.base import BaseModel, ModelInfo, resolve_device

# ----------------------------------------------------------------------------- 공통 자료형


@dataclass
class Detection:
    mask: np.ndarray  # bool H×W (원본 해상도)
    score: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (픽셀, x1/y1 포함 안 함)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ----------------------------------------------------------------------------- 사람 찾기


class PersonDetector(BaseModel):
    def detect(self, rgb: np.ndarray) -> list[Detection]:  # pragma: no cover - 인터페이스
        raise NotImplementedError


RFDETR_SEG_CLASSES = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}


def rfdetr_shape(height: int, width: int, long_side: int, multiple: int = 24) -> tuple[int, int]:
    """원본 비율을 유지한 RF-DETR 입력 크기 (높이, 너비). 둘 다 multiple 의 배수."""
    s = long_side / float(max(height, width))
    h = max(multiple, int(round(height * s / multiple)) * multiple)
    w = max(multiple, int(round(width * s / multiple)) * multiple)
    return h, w


class RFDETRSegDetector(PersonDetector):
    """RF-DETR-Seg (COCO 사전학습) 에서 person 만 꺼낸다. fp32 로 돈다 (fp16 은 객체를 들쭉날쭉 놓친다는 보고)."""

    def __init__(self, cfg: dict, size: str = "medium", threshold: float = 0.4, long_side: int = 1032):
        super().__init__(cfg)
        if size not in RFDETR_SEG_CLASSES:
            raise ValueError(f"알 수 없는 RF-DETR-Seg 크기: {size} ({list(RFDETR_SEG_CLASSES)})")
        self.size = size
        self.threshold = float(threshold)
        self.long_side = int(long_side)
        self.model = None

    def _load(self) -> None:
        try:
            import rfdetr
        except ImportError:
            raise RuntimeError("rfdetr 가 없습니다. `uv sync` 로 설치하세요.") from None
        self.model = getattr(rfdetr, RFDETR_SEG_CLASSES[self.size])()
        self._thresholds = rfdetr.ClassThresholds({"person": self.threshold}, default=float("inf"))

    def _unload(self) -> None:
        self.model = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"rfdetr_seg_{self.size}",
            weights=f"rf-detr-seg-{self.size}.pt",
            extra={"threshold": self.threshold, "long_side": self.long_side},
        )

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        self.load()
        h, w = rgb.shape[:2]
        multiple = 12 if self.size == "nano" else 24
        shape = rfdetr_shape(h, w, self.long_side, multiple)
        dets = self.model.predict(rgb, threshold=self._thresholds, shape=shape, include_source_image=False)
        out: list[Detection] = []
        if dets is None or len(dets) == 0:
            return out
        names = dets.data.get("class_name") if getattr(dets, "data", None) is not None else None
        for i in range(len(dets)):
            if names is not None and str(names[i]) != "person":
                continue
            m = np.asarray(dets.mask[i], dtype=bool) if dets.mask is not None else None
            if m is None or not m.any():
                continue
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            b = box_from_mask(m)
            out.append(Detection(mask=m, score=float(dets.confidence[i]), box=b))
        return out


class MaskFileDetector(PersonDetector):
    """모델 없이: 사용자가 준 마스크 이미지(흰색 = 사람)를 그대로 '검출' 결과로 쓴다. 사진 테스트·수동 보정용."""

    def __init__(self, cfg: dict, mask_path: str | Path):
        super().__init__(cfg)
        self.mask_path = Path(mask_path)

    def _load(self) -> None:
        pass

    def _unload(self) -> None:
        pass

    def info(self) -> ModelInfo:
        return ModelInfo(name="mask_file", weights=str(self.mask_path))

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        from vr180.util import imageio

        m = imageio.read_mask(self.mask_path)
        h, w = rgb.shape[:2]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        n, labels = cv2.connectedComponents((m > 127).astype(np.uint8))
        out = []
        for k in range(1, n):
            mk = labels == k
            out.append(Detection(mask=mk, score=1.0, box=box_from_mask(mk)))
        return out


def get_person_detector(name: str, cfg: dict, **kw) -> PersonDetector:
    name = (name or "rfdetr_seg_medium").lower()
    if name.startswith("rfdetr_seg_"):
        return RFDETRSegDetector(cfg, size=name.removeprefix("rfdetr_seg_"), **kw)
    raise ValueError(f"알 수 없는 사람 검출기: {name}")


# ----------------------------------------------------------------------------- 경계 다듬기 (매팅)


class Matter(BaseModel):
    def alpha(self, rgb: np.ndarray) -> np.ndarray:  # pragma: no cover - 인터페이스
        """RGB uint8 → float32 0..1 (같은 크기)."""
        raise NotImplementedError


class BiRefNetMatter(Matter):
    """BiRefNet (Hugging Face, trust_remote_code). 1024×1024 로 줄여 추론, 출력은 sigmoid 를 거친 0..1."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, cfg: dict, repo: str | None = None, res: int = 1024):
        super().__init__(cfg)
        self.repo = repo or cfg.get("birefnet_matting_id", "ZhengPeng7/BiRefNet-matting")
        self.res = int(res)
        self.device = resolve_device(cfg.get("device", "auto"))
        self.model = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageSegmentation

        paths.ensure_hf_env()
        torch.set_float32_matmul_precision("high")
        model = AutoModelForImageSegmentation.from_pretrained(self.repo, trust_remote_code=True)
        model = model.to(self.device).eval()
        if self.device == "cuda":
            model = model.half()
        self.model = model

    def _unload(self) -> None:
        self.model = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="birefnet_matting", weights=self.repo, device=self.device, extra={"res": self.res}
        )

    def alpha(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        self.load()
        h, w = rgb.shape[:2]
        x = cv2.resize(
            rgb,
            (self.res, self.res),
            interpolation=cv2.INTER_AREA if max(h, w) > self.res else cv2.INTER_CUBIC,
        )
        x = (x.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.device)
        if self.device == "cuda":
            t = t.half()
        with torch.inference_mode():
            pred = self.model(t)[-1].sigmoid().float()[0, 0].cpu().numpy()
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def get_matter(name: str, cfg: dict) -> Matter | None:
    """'birefnet_matting' | 'none' (모델 없이 마스크 경계를 부드럽게만)."""
    name = (name or "birefnet_matting").lower()
    if name == "none":
        return None
    if name == "birefnet_matting":
        return BiRefNetMatter(cfg)
    if name == "birefnet_portrait":
        return BiRefNetMatter(cfg, repo=cfg.get("birefnet_portrait_id", "ZhengPeng7/BiRefNet-portrait"))
    raise ValueError(f"알 수 없는 매팅 모델: {name}")


# ----------------------------------------------------------------------------- SAM 2.1 영상 추적

SAM2_SIZES = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
SAM2_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{file}"
SAM2_REPO = "https://github.com/facebookresearch/sam2.git"
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"  # 2026-11 확인한 main


def sam2_repo_dir() -> Path:
    return paths.third_party_dir() / "sam2"


def sam2_ckpt_path(size: str) -> Path:
    return paths.models_dir() / "sam2" / SAM2_SIZES[size][1]


def sam2_available(size: str = "small") -> tuple[bool, str]:
    if not (sam2_repo_dir() / "sam2" / "build_sam.py").exists():
        return False, f"SAM 2 코드 없음: {sam2_repo_dir()} (`vr180 fetch-models --person`)"
    if not sam2_ckpt_path(size).exists():
        return False, f"SAM 2.1 가중치 없음: {sam2_ckpt_path(size)} (`vr180 fetch-models --person`)"
    return True, ""


def load_sam2_video_predictor(cfg: dict, size: str = "small"):
    """third_party/sam2 를 sys.path 에 넣고 영상 추적기를 만든다. CUDA 확장(_C) 이 없으면 SAM 2 가
    작은 구멍 메우기 후처리만 건너뛴다 (결과는 거의 같음)."""
    ok, reason = sam2_available(size)
    if not ok:
        raise RuntimeError(reason)
    repo = str(sam2_repo_dir())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore

    config, _ = SAM2_SIZES[size]
    device = resolve_device(cfg.get("device", "auto"))
    return build_sam2_video_predictor(config, str(sam2_ckpt_path(size)), device=device)
