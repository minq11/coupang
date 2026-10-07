"""Stage 2 — 원근(핀홀) → half-equirect 재투영. 원본 밖은 알파 0."""

from __future__ import annotations

import math

import numpy as np

from vr180.geometry import reproject
from vr180.util import imageio

INDEX = 2
NAME = "reproj"
DIRNAME = "02_reproj"
PARAMS_KEY = "stage_02_reproj"
USES_MODELS = False


def inputs(job):
    return [job.path(0, "L.png"), job.path(0, "R.png"), job.path(1, "camera.json")]


def outputs(job):
    d = job.stage_dir(INDEX)
    return [d / "L_equi.png", d / "R_equi.png", d / "valid_mask.png", d / "preview_sbs.png"]


def run(job, params: dict, ctx) -> dict:
    cam = imageio.read_json(job.path(1, "camera.json"))
    size = int(params.get("size", 4096))
    interp = params.get("interpolation", "lanczos")
    left = imageio.read_rgb(job.path(0, "L.png"))
    right = imageio.read_rgb(job.path(0, "R.png"))
    h, w = left.shape[:2]
    hfov = math.radians(float(cam["hfov_deg"]))
    ctx.progress(0.1, "remap 테이블")
    mx, my, valid = reproject.pinhole_to_equi_maps(size, w, h, hfov)
    ctx.progress(0.3, "L 재투영")
    l_equi = reproject.remap_to_equi(left, mx, my, valid, interp)
    ctx.progress(0.6, "R 재투영")
    r_equi = reproject.remap_to_equi(right, mx, my, valid, interp)
    d = job.stage_dir(INDEX)
    imageio.write_image(d / "L_equi.png", l_equi)
    imageio.write_image(d / "R_equi.png", r_equi)
    imageio.write_image(d / "valid_mask.png", (valid.astype(np.uint8) * 255))
    bg = params.get("preview_only_background", "black")
    sbs = np.concatenate([imageio.flatten_rgba(l_equi, bg), imageio.flatten_rgba(r_equi, bg)], axis=1)
    imageio.write_image(d / "preview_sbs.png", sbs)
    cov = float(valid.mean())
    ctx.log(f"size={size} 유효 영역 {cov * 100:.1f}%")
    return {"size": size, "valid_fraction": cov}
