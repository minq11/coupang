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


SMALL = {
    "stage_02_reproj": {"size": 256},
    "stage_03_outpaint": {"tile_res": 128, "feather_px": 16, "inpainter": "opencv"},
    "stage_04_stereo": {"matcher": "sgbm", "max_disparity": 64},
    "stage_05_depth": {"estimator": "none"},
    "stage_06_right": {"inpainter": "opencv"},
}


@pytest.fixture
def small_params():
    return {k: dict(v) for k, v in SMALL.items()}


def _unused():
    return os
