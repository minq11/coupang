"""Stage 7 — pano_L | pano_R 를 붙여 VR180 SBS 출력 (PNG + JPG q95, 선택 XMP GPano).

JPG 는 Job 폴더뿐 아니라 공용 `work/output/<job_id><suffix>.jpg` 로도 복사된다 (쉽게 모아 보기용)."""

from __future__ import annotations

import shutil

import numpy as np

from vr180 import paths
from vr180.util import imageio
from vr180.util.xmp import inject_xmp_jpeg

INDEX = 7
NAME = "output"
DIRNAME = "07_output"
PARAMS_KEY = "stage_07_output"
USES_MODELS = False


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
    suffix = p.get("suffix", "_180_3dh") or ""
    out = [d / f"output_VR180_SBS{suffix}.png", d / "preview_original_only.png"]
    if p.get("write_jpg", True):
        out.append(d / f"output_VR180_SBS{suffix}.jpg")
    return out


def run(job, params: dict, ctx) -> dict:
    pl_stage, pl_name = _pano_l_source(job)
    pano_l = imageio.read_rgba(job.path(pl_stage, pl_name))[..., :3]
    pano_r = imageio.read_rgba(job.path(6, "pano_R.png"))[..., :3]
    size = pano_l.shape[0]
    sbs = np.concatenate([pano_l, pano_r], axis=1)
    suffix = params.get("suffix", "_180_3dh") or ""
    d = job.stage_dir(INDEX)
    png = d / f"output_VR180_SBS{suffix}.png"
    imageio.write_image(png, sbs)
    written = [str(png)]
    if params.get("write_jpg", True):
        jpg = d / f"output_VR180_SBS{suffix}.jpg"
        imageio.write_image(jpg, sbs, jpg_quality=int(params.get("jpg_quality", 95)))
        if params.get("write_xmp", True):
            inject_xmp_jpeg(jpg, pano_size=size)
        written.append(str(jpg))
        pool = paths.work_dir() / "output"
        pool.mkdir(parents=True, exist_ok=True)
        pool_jpg = pool / f"{job.id}{suffix}.jpg"
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
