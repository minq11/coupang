"""Video Depth Anything (VDA) — 영상 전용 시간축 깊이. 32프레임 묶음을 겹쳐 추정해 프레임 간 떨림을 없앤다.

VDA 저장소: https://github.com/DepthAnything/Video-Depth-Anything.
설치: `uv run vr180 fetch-models --vda` → third_party/Video-Depth-Anything + models/vda/*.pth.
없으면 DA2 (프레임별) 로 폴백 — 떨림이 생기지만 pipeline 자체는 돈다.

인터페이스
----------
    class VideoDepthEstimator:
        def inverse_depth_sequence(frames: list[np.ndarray]) -> list[np.ndarray]

VDA 출력은 'disparity-like' (상대 inverse depth, 클수록 가까움) → 우리 interface 와 그대로 일치.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from vr180 import paths
from vr180.models.base import ModelInfo, resolve_device


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


# ----------------------------------------------------------------------------- VDA 네이티브
_VDA_MODEL_CONFIGS = {
    "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _vda_repo_dir() -> Path:
    return paths.third_party_dir() / "Video-Depth-Anything"


def _vda_weights_path(variant: str) -> Path:
    name = f"video_depth_anything_vits.pth" if variant == "small" else "video_depth_anything_vitl.pth"
    return paths.models_dir() / "vda" / name


def _vda_available() -> tuple[bool, str]:
    """저장소·가중치 체크. (ok, 사유)."""
    repo = _vda_repo_dir()
    if not (repo / "video_depth_anything").exists():
        return False, f"VDA 저장소 없음: {repo} (vr180 fetch-models --vda 로 clone)"
    return True, ""


class VDAEstimator(VideoDepthEstimator):
    """VDA 네이티브 어댑터. infer_video_depth 를 그대로 쓰고 내부 window 에 위임."""

    def __init__(self, model_cfg: dict, variant: str = "large", chunk: int = 32, overlap: int = 8):
        self.cfg = model_cfg
        self.variant = variant  # 'small' | 'large'
        self.chunk = int(chunk)
        self.overlap = int(overlap)
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.model = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"video_depth_anything_{self.variant}",
            weights=str(_vda_weights_path(self.variant)),
            device=self.device,
            extra={"chunk": self.chunk, "overlap": self.overlap},
        )

    def _load(self) -> None:
        import torch

        ok, reason = _vda_available()
        if not ok:
            raise RuntimeError(reason)
        wp = _vda_weights_path(self.variant)
        if not wp.exists():
            raise RuntimeError(
                f"VDA 가중치 없음: {wp} (vr180 fetch-models --vda 로 다운로드)"
            )
        repo = _vda_repo_dir()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from video_depth_anything.video_depth import VideoDepthAnything  # type: ignore

        cfg = _VDA_MODEL_CONFIGS[self.variant]
        self.model = VideoDepthAnything(**cfg)
        state = torch.load(str(wp), map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()

    def _unload(self) -> None:
        self.model = None
        from vr180.models.base import free_cuda
        free_cuda()

    def __enter__(self):
        if self.model is None:
            self._load()
        return self

    def __exit__(self, *exc):
        self._unload()
        return False

    def inverse_depth_sequence(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """frames: list of RGB uint8 (H,W,3). 반환: list of float32 inverse depth (H,W)."""
        if self.model is None:
            self._load()
        if not frames:
            return []
        # VDA 는 (N,H,W,3) numpy 배열 또는 list 를 받고, 내부적으로 chunk/overlap window 를 돌림.
        # input_size=518 가 공식 기본. fp32 는 메모리 넉넉하면 True 가 조금 더 안정적.
        import torch

        with torch.inference_mode():
            # VDA 의 infer_video_depth 는 보통 (N,H,W) float32 depth 배열을 돌려준다.
            # 저장소 레이아웃에 따라 (depths, fps) 튜플인 경우도 있어 분기.
            arr = np.ascontiguousarray(np.stack(frames, axis=0))
            result = self.model.infer_video_depth(
                arr,
                target_fps=-1,  # 재샘플 안 함 (우리 쪽에서 이미 frame 리스트)
                input_size=518,
                device=self.device,
                fp32=False,
            )
            if isinstance(result, tuple):
                depths = result[0]
            else:
                depths = result
            depths_np = depths.detach().cpu().numpy() if hasattr(depths, "detach") else np.asarray(depths)
        # VDA 는 'disparity-like' (크면 가까움) → 그대로 inverse depth. 음수 방지만.
        return [np.ascontiguousarray(d.astype(np.float32)) for d in np.clip(depths_np, 0, None)]


# ----------------------------------------------------------------------------- DA2 폴백
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
        ok, reason = _vda_available()
        if ok and _vda_weights_path(variant).exists():
            return VDAEstimator(model_cfg, variant=variant, chunk=chunk, overlap=overlap)
        # VDA 없으면 자동 폴백 (일회성 안내 로그만)
        print(f"[vda] 폴백 → DA2: {reason or '가중치 없음'}")
        return DA2FallbackEstimator(model_cfg, variant=variant)
    if name in ("da2_video", "da2_video_large", "da2_video_small"):
        variant = "small" if name.endswith("small") else "large"
        return DA2FallbackEstimator(model_cfg, variant=variant)
    raise ValueError(f"알 수 없는 video depth estimator: {name!r}")
