"""모델 없는 빠른 inpainter (stretch) 와 registry 메타데이터 테스트."""

import numpy as np
import pytest

from vr180.models import registry
from vr180.models.inpaint_stretch import StretchBlurInpainter
from vr180.util.fill import nearest_fill_rgb


def _tile(size=256):
    """왼쪽 절반은 질감 있는 원본, 오른쪽 절반은 채울 구멍."""
    yy, xx = np.mgrid[0:size, 0:size]
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[..., 0] = ((xx // 8 + yy // 8) % 2) * 120 + 60  # 체커
    rgb[..., 1] = (yy * 255 // size).astype(np.uint8)  # 세로 그라데이션
    rgb[..., 2] = 200
    mask = np.zeros((size, size), np.uint8)
    mask[:, size // 2 :] = 255
    rgb[mask > 0] = 0
    return rgb, mask


def test_nearest_fill_rgb_copies_edge():
    rgb, mask = _tile(64)
    out = nearest_fill_rgb(rgb, mask == 0)
    # 구멍의 각 행은 경계 열(31) 의 값을 그대로 복사한다
    assert np.array_equal(out[:, 32:], np.repeat(rgb[:, 31:32], 32, axis=1))
    assert np.array_equal(out[:, :32], rgb[:, :32])


def test_stretch_fills_and_keeps_valid():
    rgb, mask = _tile(256)
    inp = StretchBlurInpainter({})
    with inp:
        out = inp.inpaint(rgb, mask)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    # 원본 영역은 바이트 단위로 그대로
    assert np.array_equal(out[mask == 0], rgb[mask == 0])
    # 구멍은 채워짐 (검정 아님)
    assert out[mask > 0].mean() > 30
    # 경계 바로 바깥 색은 경계 색과 비슷 (세로 그라데이션 채널)
    assert np.abs(out[:, 132, 1].astype(int) - rgb[:, 127, 1].astype(int)).mean() < 25
    # 멀수록 더 흐림: 체커 채널의 행 방향 분산이 경계 근처 > 먼 곳
    near = out[:, 130:150, 0].astype(float).std(axis=0).mean()
    far = out[:, 230:250, 0].astype(float).std(axis=0).mean()
    assert far < near
    # 멀수록 어두움 (darken 기본 0.15)
    assert out[:, 240:, :].mean() < out[:, 130:140, :].mean()


def test_stretch_no_mask_is_identity_and_darken_zero():
    rgb, mask = _tile(64)
    inp = StretchBlurInpainter({"stretch_darken": 0.0})
    assert np.array_equal(inp.inpaint(rgb, np.zeros_like(mask)), rgb)
    out = inp.inpaint(rgb, mask)
    # darken 0 이면 먼 곳 밝기가 경계 근처와 비슷 (블러만)
    assert abs(out[:, 60:, 2].mean() - out[:, 33:36, 2].mean()) < 6


def test_registry_choices_and_flags():
    names = registry.INPAINTER_NAMES
    for n in ("opencv", "stretch", "lama", "sdxl_lightning", "sdxl_lcm", "sdxl", "flux"):
        assert n in names
    assert registry.inpainter_needs_prompt("stretch") is False
    assert registry.inpainter_needs_prompt("lama") is False
    assert registry.inpainter_needs_prompt("sdxl_lcm") is True
    st = registry.get_inpainter("stretch", {})
    assert st.MODEL_FREE and not st.NEEDS_PROMPT
    assert st.info().name == "stretch_blur"
    # lama 는 생성만으로는 torch 를 import 하지 않는다 (가벼운 객체)
    lm = registry.get_inpainter("lama", {})
    assert lm.info().name == "lama_big" and not lm.NEEDS_PROMPT


def test_lama_pad_and_crop_with_fake_model():
    """TorchScript 모델 대신 '입력을 그대로 돌려주는' 가짜 모델로 패딩·크롭·마스크 합성 경로를 검증."""
    torch = pytest.importorskip("torch")
    from vr180.models.inpaint_lama import LaMaInpainter, _pad_mod8, run_lama

    a = np.zeros((3, 13, 21), np.float32)
    assert _pad_mod8(a).shape == (3, 16, 24)

    class Fake:
        def __call__(self, img, msk):
            return img * (1 - msk) + 0.5 * msk  # 구멍을 회색으로

    rgb = np.full((13, 21, 3), 200, np.uint8)
    mask = np.zeros((13, 21), np.uint8)
    mask[3:8, 4:15] = 255
    out = run_lama(Fake(), rgb, mask, "cpu")
    assert out.shape == rgb.shape
    assert (out[mask == 0] == 200).all()
    assert abs(int(out[5, 8, 0]) - 127) <= 1
    lm = LaMaInpainter({"device": "cpu"})
    lm.model = Fake()
    lm._loaded = True
    res = lm.inpaint(rgb, mask)
    assert (res[mask == 0] == 200).all() and abs(int(res[5, 8, 1]) - 127) <= 1
    del torch


def test_sdxl_variant_defaults():
    from vr180.models.inpaint_sdxl import SDXLInpainter

    base = SDXLInpainter({}, "base")
    assert base._effective_steps_guidance(30, 7.0) == (30, 7.0)
    lcm = SDXLInpainter({}, "lcm")
    assert lcm._effective_steps_guidance(30, 7.0) == (8, 1.5)
    assert lcm._effective_steps_guidance(6, 1.2) == (6, 1.2)
    lt = SDXLInpainter({}, "lightning")
    assert lt._effective_steps_guidance(30, 7.0) == (4, 0.0)
    assert lt.info().name == "sdxl_inpaint_lightning"
    with pytest.raises(ValueError):
        SDXLInpainter({}, "turbo")


def test_end_to_end_with_stretch(work_dir, sample_mono, small_params_mono):
    from vr180.pipeline.job import Job
    from vr180.util import imageio
    from vr180.util.log import RunContext

    p = small_params_mono
    p["stage_03_outpaint"]["inpainter"] = "stretch"
    p["stage_06_right"]["inpainter"] = "stretch"
    job = Job.create(sample_mono, job_id="t_stretch", params_override=p, input_mode="mono")
    job.run(ctx=RunContext(log_fn=lambda s: None))
    pano = imageio.read_rgba(job.path(3, "pano_C.png"))
    assert (pano[..., 3] == 255).all()
    assert job.meta(3)["model"]["name"] == "stretch_blur"
    assert job.meta(6)["model"]["name"] == "stretch_blur"
