"""더미(모델 없음) end-to-end: 캐시, from-stage 재실행, 외부 수정본 반영."""

import numpy as np

from vr180.pipeline.job import Job, stages
from vr180.util import imageio
from vr180.util.log import RunContext


def _ctx():
    return RunContext(log_fn=lambda s: None)


def test_end_to_end_and_cache(work_dir, sample_sbs, small_params):
    job = Job.create(sample_sbs, job_id="t1", params_override=small_params)
    res = job.run(ctx=_ctx())
    assert res == ["ran"] * len(stages())
    assert job.all_status() == ["done"] * len(stages())
    out = job.path(7, "output_VR180_SBS_180_3dh.png")
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
    job = Job.create(sample_sbs, job_id="t2", params_override=small_params)
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
    job = Job.create(sample_sbs, job_id="t3", params_override=small_params)
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
    final = imageio.read_rgb(job.path(7, "output_VR180_SBS_180_3dh.png"))
    assert final[0, 0, 0] == 255
