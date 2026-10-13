"""Stage 2 — 원근(핀홀) → half-equirect 재투영. 원본 밖은 알파 0."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vr180.geometry import extent as _extent
from vr180.geometry import reproject
from vr180.util import imageio

INDEX = 2
NAME = "reproj"
DIRNAME = "02_reproj"
PARAMS_KEY = "stage_02_reproj"
USES_MODELS = False


def inputs(job):
    if job.input_mode == "mono":
        return [job.path(0, "C.png"), job.path(1, "camera.json")]
    return [job.path(0, "L.png"), job.path(0, "R.png"), job.path(1, "camera.json")]


def outputs(job):
    d = job.stage_dir(INDEX)
    extent_outs = [d / "extent_mask.png", d / "extent_fade.png"]
    if job.input_mode == "mono":
        return [d / "C_equi.png", d / "valid_mask.png", d / "preview_sbs.png", *extent_outs]
    return [d / "L_equi.png", d / "R_equi.png", d / "valid_mask.png", d / "preview_sbs.png", *extent_outs]


def _write_extent(size: int, cam: dict, params: dict, d, ctx) -> dict:
    """extent_mask.png / extent_fade.png 를 쓰고 요약을 돌려준다."""
    hfov = float(cam.get("hfov_deg", 80.0))
    vfov = float(cam.get("vfov_deg", 60.0))
    h_deg, v_deg = _extent.compute_extent_hv(hfov, vfov, params)
    fade_deg = float(params.get("extent_fade_deg", 10.0))
    shape = params.get("extent_shape", "ellipse")
    _, w = _extent.extent_weight(size, h_deg, v_deg, fade_deg, shape)
    mask = (w > 0).astype(np.uint8) * 255
    fade_u8 = (np.clip(w, 0, 1) * 255).astype(np.uint8)
    imageio.write_image(d / "extent_mask.png", mask)
    imageio.write_image(d / "extent_fade.png", fade_u8)
    ctx.log(f"extent {h_deg:.1f}°×{v_deg:.1f}° fade={fade_deg:.1f}° shape={shape}")
    return {
        "extent_h_deg": h_deg,
        "extent_v_deg": v_deg,
        "extent_fade_deg": fade_deg,
        "extent_shape": shape,
        "extent_inside_fraction": float((w > 0).mean()),
    }


def _draw_extent_boundary(preview_sbs: np.ndarray, size: int, extent_info: dict) -> np.ndarray:
    """preview_sbs (높이 size, 폭 2·size) 양쪽 눈에 r=1 경계를 1px 흰 선으로."""
    img = preview_sbs.copy()
    h_deg = extent_info["extent_h_deg"]
    v_deg = extent_info["extent_v_deg"]
    if _extent.is_disabled(h_deg, v_deg):
        return img
    # equirect: 중앙 (θ=0, φ=0) 은 (size/2, size/2). deg_per_pixel = 180/size.
    dpp = 180.0 / size
    cx, cy = size // 2, size // 2
    axis_x = int(round((h_deg / 2.0) / dpp))
    axis_y = int(round((v_deg / 2.0) / dpp))
    for ox in (0, size):  # 양쪽 눈
        cv2.ellipse(img, (cx + ox, cy), (axis_x, axis_y), 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run(job, params: dict, ctx) -> dict:
    cam = imageio.read_json(job.path(1, "camera.json"))
    size = int(params.get("size", 4096))
    interp = params.get("interpolation", "lanczos")
    d = job.stage_dir(INDEX)
    bg = params.get("preview_only_background", "black")

    pitch = math.radians(float(params.get("pitch_offset_deg", 0.0)))

    if job.input_mode == "mono":
        center = imageio.read_rgb(job.path(0, "C.png"))
        h, w = center.shape[:2]
        hfov = math.radians(float(cam["hfov_deg"]))
        ctx.progress(0.2, "remap 테이블")
        mx, my, valid = reproject.pinhole_to_equi_maps(size, w, h, hfov, pitch)
        ctx.progress(0.5, "C 재투영")
        c_equi = reproject.remap_to_equi(center, mx, my, valid, interp)
        imageio.write_image(d / "C_equi.png", c_equi)
        imageio.write_image(d / "valid_mask.png", (valid.astype(np.uint8) * 255))
        ex = _write_extent(size, cam, params, d, ctx)
        flat = imageio.flatten_rgba(c_equi, bg)
        sbs = np.concatenate([flat, flat], axis=1)
        sbs = _draw_extent_boundary(sbs, size, ex)
        imageio.write_image(d / "preview_sbs.png", sbs)
        cov = float(valid.mean())
        ctx.log(f"mono size={size} 유효 영역 {cov * 100:.1f}%")
        return {"size": size, "valid_fraction": cov, "mode": "mono", **ex}

    left = imageio.read_rgb(job.path(0, "L.png"))
    right = imageio.read_rgb(job.path(0, "R.png"))
    h, w = left.shape[:2]
    hfov = math.radians(float(cam["hfov_deg"]))
    ctx.progress(0.1, "remap 테이블")
    mx, my, valid = reproject.pinhole_to_equi_maps(size, w, h, hfov, pitch)
    ctx.progress(0.3, "L 재투영")
    l_equi = reproject.remap_to_equi(left, mx, my, valid, interp)
    ctx.progress(0.6, "R 재투영")
    r_equi = reproject.remap_to_equi(right, mx, my, valid, interp)
    imageio.write_image(d / "L_equi.png", l_equi)
    imageio.write_image(d / "R_equi.png", r_equi)
    imageio.write_image(d / "valid_mask.png", (valid.astype(np.uint8) * 255))
    ex = _write_extent(size, cam, params, d, ctx)
    sbs = np.concatenate([imageio.flatten_rgba(l_equi, bg), imageio.flatten_rgba(r_equi, bg)], axis=1)
    sbs = _draw_extent_boundary(sbs, size, ex)
    imageio.write_image(d / "preview_sbs.png", sbs)
    cov = float(valid.mean())
    ctx.log(f"size={size} 유효 영역 {cov * 100:.1f}%")
    return {"size": size, "valid_fraction": cov, **ex}
