"""Stage 7 — pano_L | pano_R 를 붙여 VR180 SBS 출력 (PNG + JPG q95, 선택 XMP GPano).

JPG 는 Job 폴더뿐 아니라 공용 `work/output/<job_id><tag><suffix>.jpg` 로도 복사된다 (쉽게 모아 보기용).

파일명은 핵심 옵션값을 '__' 로 묶어 태그로 넣어서, 옵션을 바꿔 재실행해도 이전 결과를
덮어쓰지 않는다. option_tag() 가 GUI/탭 로더도 공유함."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np

from vr180 import paths
from vr180.util import imageio
from vr180.util.xmp import inject_xmp_jpeg

if TYPE_CHECKING:
    from vr180.pipeline.job import Job

INDEX = 7
NAME = "output"
DIRNAME = "07_output"
PARAMS_KEY = "stage_07_output"
USES_MODELS = False


def option_tag(job: "Job") -> str:
    """핵심 옵션값을 '__' 로 구분한 파일명 태그. 선두에도 '__' 가 붙어 suffix 와 바로 이어진다.

    예 (mono): '__h-60__s-4096__e-180x180__f-10__ip-opencv__dm-da2_large__ds-0.02'
    """
    p = job.params
    s1 = p.get("stage_01_fov", {})
    s2 = p.get("stage_02_reproj", {})
    s3 = p.get("stage_03_outpaint", {})
    s5 = p.get("stage_05_depth", {})
    parts = [
        f"h-{float(s1.get('hfov_deg', 80.0)):g}",
        f"s-{int(s2.get('size', 4096))}",
        f"e-{float(s2.get('extent_h_deg', 180.0)):g}x{float(s2.get('extent_v_deg', 180.0)):g}",
        f"f-{float(s2.get('extent_fade_deg', 10.0)):g}",
        f"ip-{s3.get('inpainter', 'opencv')}",
        f"dm-{s5.get('estimator', 'da2_large')}",
    ]
    if job.input_mode == "mono":
        parts.append(f"ds-{float(s5.get('depth_strength', 0.02)):g}")
    return "__" + "__".join(parts)


def output_basename(job: "Job") -> str:
    """출력 파일 베이스네임 (확장자 제외): 'output_VR180_SBS{tag}{suffix}'."""
    tag = option_tag(job)
    suffix = (job.params.get(PARAMS_KEY, {}).get("suffix", "_180_3dh") or "")
    return f"output_VR180_SBS{tag}{suffix}"


def _pano_l_source(job) -> tuple[int, str]:
    """mono: 양쪽 다 Stage 6 가 만든다. sbs: pano_L 은 Stage 3 에서 바로 사용."""
    return (6, "pano_L.png") if job.input_mode == "mono" else (3, "pano_L.png")


def inputs(job):
    pl_stage, pl_name = _pano_l_source(job)
    if job.input_mode == "mono":
        return [
            job.path(pl_stage, pl_name),
            job.path(6, "pano_R.png"),
            job.path(2, "C_equi.png"),
        ]
    return [
        job.path(pl_stage, pl_name),
        job.path(6, "pano_R.png"),
        job.path(2, "L_equi.png"),
        job.path(2, "R_equi.png"),
    ]


def outputs(job):
    d = job.stage_dir(INDEX)
    p = job.params.get(PARAMS_KEY, {})
    base = output_basename(job)
    out = [d / f"{base}.png", d / "preview_original_only.png"]
    if p.get("write_jpg", True):
        out.append(d / f"{base}.jpg")
    return out


def run(job, params: dict, ctx) -> dict:
    pl_stage, pl_name = _pano_l_source(job)
    pano_l = imageio.read_rgba(job.path(pl_stage, pl_name))[..., :3]
    pano_r = imageio.read_rgba(job.path(6, "pano_R.png"))[..., :3]
    size = pano_l.shape[0]
    sbs = np.concatenate([pano_l, pano_r], axis=1)
    base = output_basename(job)
    d = job.stage_dir(INDEX)
    png = d / f"{base}.png"
    imageio.write_image(png, sbs)
    written = [str(png)]
    if params.get("write_jpg", True):
        jpg = d / f"{base}.jpg"
        imageio.write_image(jpg, sbs, jpg_quality=int(params.get("jpg_quality", 95)))
        if params.get("write_xmp", True):
            inject_xmp_jpeg(jpg, pano_size=size)
        written.append(str(jpg))
        pool = paths.work_dir() / "output"
        pool.mkdir(parents=True, exist_ok=True)
        # 공용 풀에도 Job id 와 옵션 태그를 함께 넣어 중복 방지.
        tag_suffix = base[len("output_VR180_SBS"):]
        pool_jpg = pool / f"{job.id}{tag_suffix}.jpg"
        shutil.copy2(jpg, pool_jpg)
        ctx.log(f"→ 공용 출력: {pool_jpg}")
        written.append(str(pool_jpg))
    # 원본 영역만 미리보기 (sbs: L_equi|R_equi, mono: C_equi 를 양쪽에 복제)
    if job.input_mode == "mono":
        c_equi = imageio.read_rgba(job.path(2, "C_equi.png"))
        flat = imageio.flatten_rgba(c_equi)
        prev = np.concatenate([flat, flat], axis=1)
    else:
        l_equi = imageio.read_rgba(job.path(2, "L_equi.png"))
        r_equi = imageio.read_rgba(job.path(2, "R_equi.png"))
        prev = np.concatenate([imageio.flatten_rgba(l_equi), imageio.flatten_rgba(r_equi)], axis=1)
    imageio.write_image(d / "preview_original_only.png", prev)
    ctx.log("출력: " + ", ".join(written))
    return {"files": written, "width": sbs.shape[1], "height": sbs.shape[0]}
