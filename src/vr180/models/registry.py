"""이름 → 구현체. 설정 파일의 문자열 하나로 모델을 바꾼다."""

from __future__ import annotations

from vr180.models.base import DepthEstimator, FovEstimator, Inpainter, StereoMatcher

# GUI 드롭다운(사진·영상·Stage 6)·README 가 같은 목록을 쓴다. (name, 설명, 프롬프트 사용 여부)
INPAINTER_CHOICES: list[tuple[str, str, bool]] = [
    ("opencv", "Telea — 모델 없음, 번짐. 뼈대 확인용", False),
    ("stretch", "늘리기+블러 — 모델 없음, 밀리초. 색만 이어짐(앰비언트). 범위 제한과 궁합 최고", False),
    ("lama", "LaMa — 경량 신경망, ~1초. 벽·바닥 선을 이음. 첫 실행 때 가중치 200MB 자동 다운로드", False),
    ("sdxl_lightning", "SDXL + Lightning 4-step LoRA — 2~4초. 내용 생성, 프롬프트 약하게 반영", True),
    ("sdxl_lcm", "SDXL + LCM-LoRA 8-step — 3~5초. 내용 생성, 프롬프트·네거티브 반영", True),
]
INPAINTER_NAMES = [c[0] for c in INPAINTER_CHOICES]


def inpainter_needs_prompt(name: str) -> bool:
    for n, _, needs in INPAINTER_CHOICES:
        if n == (name or "opencv").lower():
            return needs
    return True


def get_inpainter(name: str, model_cfg: dict) -> Inpainter:
    name = (name or "opencv").lower()
    if name == "opencv":
        from vr180.models.inpaint_opencv import OpenCVInpainter

        return OpenCVInpainter(model_cfg)
    if name == "stretch":
        from vr180.models.inpaint_stretch import StretchBlurInpainter

        return StretchBlurInpainter(model_cfg)
    if name == "lama":
        from vr180.models.inpaint_lama import LaMaInpainter

        return LaMaInpainter(model_cfg)
    if name in ("sdxl_lcm", "sdxl_lightning"):
        from vr180.models.inpaint_sdxl import SDXLInpainter

        return SDXLInpainter(model_cfg, variant=name.removeprefix("sdxl_"))
    raise ValueError(f"알 수 없는 inpainter: {name} (가능: {', '.join(INPAINTER_NAMES)})")


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
