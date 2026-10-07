"""Stage 7 — pano_L | pano_R 를 붙여 VR180 SBS 출력 (PNG + JPG q95, 선택 XMP GPano)."""

from __future__ import annotations

import numpy as np

from vr180.util import imageio
from vr180.util.xmp import inject_xmp_jpeg

INDEX = 7
NAME = "output"
DIRNAME = "07_output"
PARAMS_KEY = "stage_07_output"
USES_MODELS = False


def inputs(job):
    return [
        job.path(3, "pano_L.png"),
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
    pano_l = imageio.read_rgba(job.path(3, "pano_L.png"))[..., :3]
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
    l_equi = imageio.read_rgba(job.path(2, "L_equi.png"))
    r_equi = imageio.read_rgba(job.path(2, "R_equi.png"))
    prev = np.concatenate([imageio.flatten_rgba(l_equi), imageio.flatten_rgba(r_equi)], axis=1)
    imageio.write_image(d / "preview_original_only.png", prev)
    ctx.log("출력: " + ", ".join(written))
    return {"files": written, "width": sbs.shape[1], "height": sbs.shape[0]}
