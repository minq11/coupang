"""모델 인터페이스. 구현체는 registry 에서 이름으로 고른다.

모든 구현체는 `load()` 전까지 무거운 import 를 하지 않고, `unload()` 로 VRAM 을 비운다.
with 문으로 쓰면 자동으로 unload 된다. 두 모델을 동시에 올리지 않는 것이 원칙.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelInfo:
    name: str
    weights: str = ""  # repo id / 파일 경로
    weights_hash: str = ""  # 재현용 (알 수 있을 때)
    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "weights": self.weights,
            "weights_hash": self.weights_hash,
            "device": self.device,
            **self.extra,
        }


class BaseModel(ABC):
    def __init__(self, model_cfg: dict):
        self.cfg = model_cfg
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def unload(self) -> None:
        if self._loaded:
            self._unload()
            self._loaded = False
        free_cuda()

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def _unload(self) -> None: ...

    @abstractmethod
    def info(self) -> ModelInfo: ...

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.unload()
        return False


class Inpainter(BaseModel):
    """RGB uint8 (H×W×3) + 마스크 uint8 (255 = 채울 곳) → RGB uint8."""

    @abstractmethod
    def inpaint(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = 0,
        steps: int = 30,
        guidance: float = 7.0,
        strength: float = 1.0,
    ) -> np.ndarray: ...


class DepthEstimator(BaseModel):
    """RGB uint8 → 상대 inverse depth float32 (H×W, 클수록 가까움). 스케일/오프셋은 임의."""

    @abstractmethod
    def inverse_depth(self, rgb: np.ndarray) -> np.ndarray: ...


class StereoMatcher(BaseModel):
    """정류된 L/R RGB uint8 → Left 기준 픽셀 disparity float32 (H×W, 양수 = 가까움), 유효 마스크 bool."""

    @abstractmethod
    def disparity(
        self, left: np.ndarray, right: np.ndarray, max_disparity: int = 256
    ) -> tuple[np.ndarray, np.ndarray]: ...


class FovEstimator(BaseModel):
    """RGB uint8 → (hfov_deg, confidence 0..1). 실패하면 (None, 0)."""

    @abstractmethod
    def estimate_hfov(self, rgb: np.ndarray) -> tuple[float | None, float]: ...


def resolve_device(cfg_device: str = "auto") -> str:
    if cfg_device and cfg_device != "auto":
        return cfg_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def free_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
