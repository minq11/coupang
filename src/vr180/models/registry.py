"""이름 → 구현체. 설정 파일의 문자열 하나로 모델을 바꾼다."""

from __future__ import annotations

from vr180.models.base import DepthEstimator, FovEstimator, Inpainter, StereoMatcher


def get_inpainter(name: str, model_cfg: dict) -> Inpainter:
    name = (name or "opencv").lower()
    if name == "opencv":
        from vr180.models.inpaint_opencv import OpenCVInpainter

        return OpenCVInpainter(model_cfg)
    if name == "sdxl":
        from vr180.models.inpaint_sdxl import SDXLInpainter

        return SDXLInpainter(model_cfg)
    if name == "flux":
        from vr180.models.inpaint_flux import FluxFillInpainter

        return FluxFillInpainter(model_cfg)
    raise ValueError(f"알 수 없는 inpainter: {name}")


def get_depth_estimator(name: str, model_cfg: dict) -> DepthEstimator | None:
    name = (name or "none").lower()
    if name == "none":
        return None
    if name.startswith("da2_"):
        from vr180.models.depth_da2 import DepthAnythingV2

        return DepthAnythingV2(model_cfg, variant=name.split("_", 1)[1])
    raise ValueError(f"알 수 없는 depth estimator: {name}")


def get_stereo_matcher(name: str, model_cfg: dict, **kw) -> StereoMatcher:
    name = (name or "sgbm").lower()
    if name == "sgbm":
        from vr180.models.stereo_sgbm import SGBMMatcher

        return SGBMMatcher(model_cfg)
    if name == "raft_stereo":
        from vr180.models.stereo_raft import RaftStereoMatcher

        return RaftStereoMatcher(model_cfg, **kw)
    if name == "raft_flow":
        from vr180.models.stereo_raft import RaftFlowMatcher

        return RaftFlowMatcher(model_cfg, **kw)
    raise ValueError(f"알 수 없는 stereo matcher: {name}")


def get_fov_estimator(name: str, model_cfg: dict) -> FovEstimator | None:
    name = (name or "manual").lower()
    if name == "manual":
        return None
    if name == "geocalib":
        from vr180.models.fov_geocalib import GeoCalibFov

        return GeoCalibFov(model_cfg)
    raise ValueError(f"알 수 없는 FOV estimator: {name}")
