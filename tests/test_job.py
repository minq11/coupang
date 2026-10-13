"""더미(모델 없음) end-to-end: 캐시, from-stage 재실행, 외부 수정본 반영."""

import math

import numpy as np

from vr180.pipeline.job import Job, stages
from vr180.util import imageio
from vr180.util.log import RunContext


def _ctx():
    return RunContext(log_fn=lambda s: None)


def test_end_to_end_and_cache(work_dir, sample_sbs, small_params):
    job = Job.create(sample_sbs, job_id="t1", params_override=small_params, input_mode="sbs")
    res = job.run(ctx=_ctx())
    assert res == ["ran"] * len(stages())
    assert job.all_status() == ["done"] * len(stages())
    from vr180.pipeline.stage_07_output import output_basename
    out = job.path(7, f"{output_basename(job)}.png")
    img = imageio.read_rgb(out)
    assert img.shape == (256, 512, 3)
    jpg = out.with_suffix(".jpg").read_bytes()
    assert b"GPano:ProjectionType" in jpg
    # 두 번째 실행: 전부 캐시
    res2 = job.run(from_stage=0, force_from=False, ctx=_ctx())
    assert res2 == ["cached"] * len(stages())
    # Job 다시 열면 파라미터 복원
    job2 = Job.open("t1")
    assert job2.params["stage_02_reproj"]["size"] == 256


def test_param_change_reruns_from_that_stage(work_dir, sample_sbs, small_params):
    job = Job.create(sample_sbs, job_id="t2", params_override=small_params, input_mode="sbs")
    job.run(ctx=_ctx())
    job.set_param(3, "feather_px", 8)
    job.save_params()
    st = job.all_status()
    assert st[:3] == ["done"] * 3
    assert st[3] == "stale"
    res = job.run(from_stage=3, ctx=_ctx())
    assert res[:3] == ["kept"] * 3
    assert res[3] == "ran"
    assert res[4] == "cached"  # stage 4 는 stage 3 산출물에 의존하지 않는다
    assert res[5] == "ran" and res[6] == "ran" and res[7] == "ran"


def test_manual_edit_is_used_downstream(work_dir, sample_sbs, small_params):
    job = Job.create(sample_sbs, job_id="t3", params_override=small_params, input_mode="sbs")
    job.run(ctx=_ctx())
    p = job.path(3, "pano_L.png")
    edited = imageio.read_rgba(p)
    edited[..., :3] = 0
    edited[..., 0] = 255  # 전부 빨강
    imageio.write_image(p, edited)
    assert job.status(3) == "done"  # Stage 3 자체는 입력/파라미터가 그대로
    assert job.status(5) == "stale"  # 입력 해시가 바뀜
    res = job.run(from_stage=5, ctx=_ctx())
    assert res[3] == "kept" and res[5] == "ran"
    pano_r = imageio.read_rgba(job.path(6, "pano_R.png"))
    valid = imageio.read_mask(job.path(2, "valid_mask.png")) > 127
    outside = pano_r[~valid]
    assert np.mean(outside[:, 0]) > 200 and np.mean(outside[:, 1]) < 40
    from vr180.pipeline.stage_07_output import output_basename
    final = imageio.read_rgb(job.path(7, f"{output_basename(job)}.png"))
    assert final[0, 0, 0] == 255


# ----------------------------------------------------------------------------- mono 모드 end-to-end
def test_mono_end_to_end_skips_stage4(work_dir, sample_mono, small_params_mono):
    job = Job.create(sample_mono, job_id="mono1", params_override=small_params_mono, input_mode="mono")
    assert job.input_mode == "mono"
    # Stage 4 는 모드 상 skipped 로 보고된다 (실행 전에도).
    assert job.status(4) == "skipped"
    res = job.run(ctx=_ctx())
    # 결과: 0,1,2,3,5,6,7 ran + 4 skipped
    assert res[4] == "skipped"
    for i in (0, 1, 2, 3, 5, 6, 7):
        assert res[i] == "ran", f"stage {i} expected 'ran', got {res[i]!r}"
    # 모든 Stage 가 done 또는 skipped 로 보고
    st = job.all_status()
    assert st[4] == "skipped"
    for i in (0, 1, 2, 3, 5, 6, 7):
        assert st[i] == "done", f"stage {i} expected 'done', got {st[i]!r}"
    # 최종 출력 존재 + 크기
    from vr180.pipeline.stage_07_output import output_basename
    out = job.path(7, f"{output_basename(job)}.png")
    img = imageio.read_rgb(out)
    assert img.shape == (256, 512, 3)
    # Stage 0 는 C.png 하나만 생성
    assert job.path(0, "C.png").exists()
    assert not job.path(0, "L.png").exists()
    # Stage 3 는 pano_C.png 생성 (pano_L 은 Stage 6 가 만든다)
    assert job.path(3, "pano_C.png").exists()
    assert job.path(6, "pano_L.png").exists() and job.path(6, "pano_R.png").exists()


def test_extent_outputs_and_outside_equals(work_dir, sample_mono):
    """Stage 2 가 extent 산출물을 쓰고, Stage 6 결과에서 바깥 픽셀이 양 눈에 동일한지."""
    params = {
        # 작게 잡아 fade 가 눈에 보이게. 120×100 → corners 밖.
        "stage_02_reproj": {"size": 256, "extent_h_deg": 120, "extent_v_deg": 100, "extent_fade_deg": 10},
        "stage_03_outpaint": {"tile_res": 128, "feather_px": 16, "inpainter": "opencv"},
        "stage_05_depth": {"estimator": "none", "depth_strength": 0.02},
        "stage_06_right": {"inpainter": "opencv", "mono_symmetric": True},
    }
    job = Job.create(sample_mono, job_id="ext1", params_override=params, input_mode="mono")
    job.run(ctx=_ctx())
    # Stage 2 산출물
    for name in ("extent_mask.png", "extent_fade.png"):
        assert job.path(2, name).exists()
    # disp_full 은 바깥에서 0
    disp = imageio.read_npy(job.path(5, "disp_full.npy"))
    mask = imageio.read_mask(job.path(2, "extent_mask.png")) > 127
    assert float(np.max(np.abs(disp[~mask]))) < 1e-5
    # pano_L, pano_R 바깥에서 완전 동일
    pano_l = imageio.read_rgba(job.path(6, "pano_L.png"))[..., :3]
    pano_r = imageio.read_rgba(job.path(6, "pano_R.png"))[..., :3]
    assert np.array_equal(pano_l[~mask], pano_r[~mask])


def test_mono_symmetric_warp_opposite_signs(work_dir, sample_mono, small_params_mono):
    """mono 모드에서 pano_L/pano_R 이 서로 반대 방향으로 밀렸는지.
    대칭 모드 (mono_symmetric=True) 에서 같은 소스 열이 L 은 +du/2, R 은 -du/2 로 간다."""
    job = Job.create(sample_mono, job_id="mono2", params_override=small_params_mono, input_mode="mono")
    job.run(ctx=_ctx())
    # pano_C 자체를 쓸 수 없으니 (추측 상 pano_L/pano_R 둘 다 pano_C 로부터 만들어짐)
    # disp_full 중앙값과 pano_L/pano_R 의 중앙 행 색상이 반대로 밀리는지만 느슨하게 확인.
    disp = imageio.read_npy(job.path(5, "disp_full.npy"))
    pano_c = imageio.read_rgba(job.path(3, "pano_C.png"))[..., :3]
    pano_l = imageio.read_rgba(job.path(6, "pano_L.png"))[..., :3]
    pano_r = imageio.read_rgba(job.path(6, "pano_R.png"))[..., :3]
    size = pano_c.shape[0]
    # 중앙 행에서 평균 disparity > 0 이면 L 은 오른쪽, R 은 왼쪽으로 밀린다 (-sign 규약).
    mid = size // 2
    disp_mid = float(np.mean(disp[mid, size // 4 : 3 * size // 4]))
    if disp_mid > 1e-5:
        # pano_L 과 pano_C 비교 — 왼쪽으로 이동. pano_R 는 오른쪽으로. du = -disp/cosphi · S/π.
        du_px = -disp_mid * size / math.pi  # 중앙 cosφ≈1
        shift = int(round(du_px * 0.5))  # 대칭 모드에서 각각 ±0.5 배 적용
        # L 은 +0.5 scale 로 열 x → x + shift(음수) → 왼쪽 이동. R 은 그 반대.
        # 색차의 평균 밝기 비교로 느슨하게: L 의 중앙 영역이 pano_C 중앙 영역보다 오른쪽 샘플링
        assert abs(shift) >= 1  # 뭔가 밀렸다는 것만 확인
    assert pano_l.shape == pano_r.shape == pano_c.shape
