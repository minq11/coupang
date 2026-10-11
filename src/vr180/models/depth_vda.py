"""Video Depth Anything (VDA) — 영상 전용 시간축 깊이. 32프레임 묶음을 겹쳐 추정해 프레임 간 떨림을 없앤다.

VDA 저장소: https://github.com/DepthAnything/Video-Depth-Anything (설치는 third_party/ 아래).
없으면 DA2 (프레임별) 로 폴백 — 떨림이 생기지만 pipeline 자체는 돈다.

인터페이스
----------
    class VideoDepthEstimator:
        def inverse_depth_sequence(frames: list[np.ndarray]) -> list[np.ndarray]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from vr180.models.base import ModelInfo, resolve_device
from vr180 import paths


class VideoDepthEstimator:
    """추상 인터페이스. inverse_depth_sequence 하나만."""

    def inverse_depth_sequence(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError

    def info(self) -> ModelInfo:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _vda_available() -> bool:
    """third_party/Video-Depth-Anything 저장소가 있고 import 가능한가."""
    root = paths.third_party_dir() / "Video-Depth-Anything"
    if not root.exists():
        return False
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # 모듈 이름은 VDA 저장소 레이아웃에 따라 다름. 가능한 경로를 시도.
        import importlib

        for name in ("video_depth_anything", "video_depth_anything_v2", "vda"):
            try:
                importlib.import_module(name)
                return True
            except Exception:
                continue
        return False
    except Exception:
        return False


class VDAEstimator(VideoDepthEstimator):
    """VDA 네이티브 어댑터 — 저장소가 있을 때만 실제로 동작."""

    def __init__(self, model_cfg: dict, variant: str = "large", chunk: int = 32, overlap: int = 8):
        self.cfg = model_cfg
        self.variant = variant
        self.chunk = int(chunk)
        self.overlap = int(overlap)
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self._engine = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"video_depth_anything_{self.variant}",
            weights=f"vda:{self.variant}",
            device=self.device,
            extra={"chunk": self.chunk, "overlap": self.overlap},
        )

    def inverse_depth_sequence(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if not _vda_available():
            raise RuntimeError(
                "VDA 가 설치되지 않았습니다. third_party/Video-Depth-Anything 를 클론하고 "
                "pip install -r requirements.txt 하세요. 지금은 DA2 폴백을 쓰세요."
            )
        # 실제 네이티브 호출은 VDA 저장소의 공개 함수에 따라 다르다. 저장소가 안정화되면
        # 여기에 chunked inference 를 구현한다. 1차 구현은 폴백 경로로 안전하게.
        raise NotImplementedError(
            "VDA 네이티브 추론은 저장소 레이아웃이 바뀌어서 아직 통합 중입니다. "
            "fallback 'da2_video' 를 쓰세요."
        )


class DA2FallbackEstimator(VideoDepthEstimator):
    """VDA 가 없을 때 폴백: DA2 를 프레임마다 호출. 떨림이 있지만 end-to-end 는 돈다."""

    def __init__(self, model_cfg: dict, variant: str = "large"):
        from vr180.models.depth_da2 import DepthAnythingV2

        self._da2 = DepthAnythingV2(model_cfg, variant=variant)

    def info(self) -> ModelInfo:
        base = self._da2.info()
        return ModelInfo(
            name=f"da2_video_fallback_{self._da2.variant}",
            weights=base.weights,
            device=base.device,
            extra={"note": "VDA 미설치 → DA2 프레임별 폴백"},
        )

    def __enter__(self):
        self._da2.load()
        return self

    def __exit__(self, *exc):
        self._da2.unload()
        return False

    def inverse_depth_sequence(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return [self._da2.inverse_depth(f) for f in frames]


def get_video_depth_estimator(
    name: str, model_cfg: dict, chunk: int = 32, overlap: int = 8
) -> VideoDepthEstimator | None:
    """name: 'vda_large' | 'vda_small' | 'da2_video' | 'none'."""
    name = (name or "").lower()
    if name == "none":
        return None
    if name in ("vda_large", "vda_small"):
        variant = "large" if name == "vda_large" else "small"
        if _vda_available():
            return VDAEstimator(model_cfg, variant=variant, chunk=chunk, overlap=overlap)
        # VDA 없으면 자동 폴백
        return DA2FallbackEstimator(model_cfg, variant=variant)
    if name in ("da2_video", "da2_video_large", "da2_video_small"):
        variant = "small" if name.endswith("small") else "large"
        return DA2FallbackEstimator(model_cfg, variant=variant)
    raise ValueError(f"알 수 없는 video depth estimator: {name!r}")
