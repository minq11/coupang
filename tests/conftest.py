import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VR180_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("VR180_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path / "work"


@pytest.fixture
def sample_sbs(tmp_path):
    from vr180.sample import make_sbs
    from vr180.util import imageio

    p = tmp_path / "sbs.png"
    imageio.write_image(p, make_sbs(480, 270, 80.0))
    return p


@pytest.fixture
def sample_mono(tmp_path):
    from vr180.sample import make_mono
    from vr180.util import imageio

    p = tmp_path / "mono.png"
    imageio.write_image(p, make_mono(480, 270, 80.0))
    return p


SMALL = {
    # extent 180×180 = 기능 꺼짐 (레거시 테스트는 전체 180° 생성 가정).
    "stage_02_reproj": {"size": 256, "extent_h_deg": 180, "extent_v_deg": 180, "extent_fade_deg": 0},
    "stage_03_outpaint": {"tile_res": 128, "feather_px": 16, "inpainter": "opencv"},
    "stage_04_stereo": {"matcher": "sgbm", "max_disparity": 64},
    "stage_05_depth": {"estimator": "none"},
    "stage_06_right": {"inpainter": "opencv"},
}


@pytest.fixture
def small_params():
    return {k: dict(v) for k, v in SMALL.items()}


@pytest.fixture
def small_params_mono():
    """mono 모드용 작은 파라미터 — Stage 4 는 자동 skipped, estimator=none 으로 모델 없음.
    extent 는 180 으로 꺼서 레거시 어서션(원본 외 영역)이 그대로 유효하게."""
    return {
        "stage_02_reproj": {"size": 256, "extent_h_deg": 180, "extent_v_deg": 180, "extent_fade_deg": 0},
        "stage_03_outpaint": {"tile_res": 128, "feather_px": 16, "inpainter": "opencv"},
        "stage_05_depth": {"estimator": "none", "depth_strength": 0.02},
        "stage_06_right": {"inpainter": "opencv", "mono_symmetric": True},
    }


def _unused():
    return os
