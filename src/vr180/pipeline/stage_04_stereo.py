"""Stage 4 — 중앙 stereo disparity (원근 L/R 에서) → 각도 disparity 로 equirect 에 투영."""

from __future__ import annotations

import math

import numpy as np

from vr180.geometry import reproject, sphere
from vr180.util import imageio

INDEX = 4
NAME = "stereo"
DIRNAME = "04_stereo"
PARAMS_KEY = "stage_04_stereo"
USES_MODELS = True


def inputs(job):
    return [
        job.path(0, "L.png"),
        job.path(0, "R.png"),
        job.path(1, "camera.json"),
        job.path(2, "valid_mask.png"),
    ]


def outputs(job):
    d = job.stage_dir(INDEX)
    return [
        d / "disp_center.npy",
        d / "disp_center_equi.npy",
        d / "disp_center_preview.png",
        d / "disp_info.json",
    ]


def run(job, params: dict, ctx) -> dict:
    from vr180.models.registry import get_stereo_matcher

    left = imageio.read_rgb(job.path(0, "L.png"))
    right = imageio.read_rgb(job.path(0, "R.png"))
    cam = imageio.read_json(job.path(1, "camera.json"))
    valid_equi = imageio.read_mask(job.path(2, "valid_mask.png")) > 127
    size = valid_equi.shape[0]
    h, w = left.shape[:2]
    f = float(cam["focal_px"])
    hfov = math.radians(float(cam["hfov_deg"]))

    matcher = get_stereo_matcher(
        params.get("matcher", "sgbm"),
        job.model_params(),
        iters=int(params.get("raft_iters", 32)),
        weights=params.get("raft_weights", "middlebury"),
    )
    info = matcher.info()
    ctx.log(f"stereo={info.name} {w}x{h}")
    ctx.progress(0.1, "stereo matching")
    with matcher:
        disp_px, valid_px = matcher.disparity(left, right, int(params.get("max_disparity", 256)))
    disp_px = np.clip(disp_px, 0, float(params.get("max_disparity", 256))).astype(np.float32)

    # 픽셀 → 각도 disparity
    xs = (np.arange(w, dtype=np.float32) + 0.5 - w / 2.0) / f
    x_norm = np.broadcast_to(xs[None, :], (h, w))
    disp_rad = sphere.pixel_disparity_to_angular(
        disp_px, x_norm, f, exact=bool(params.get("angular_exact", True))
    )

    ctx.progress(0.6, "equirect 투영")
    mx, my, inside = reproject.pinhole_to_equi_maps(size, w, h, hfov)
    disp_equi = reproject.remap_float_to_equi(disp_rad, mx, my, inside & valid_equi, fill=0.0)

    d = job.stage_dir(INDEX)
    imageio.write_npy(d / "disp_center.npy", disp_px)
    imageio.write_npy(d / "disp_center_equi.npy", disp_equi)
    prev, rng = imageio.colormap_preview(disp_equi, valid_equi)
    imageio.write_image(d / "disp_center_preview.png", prev)
    prev_px, rng_px = imageio.colormap_preview(disp_px, valid_px)
    imageio.write_image(d / "disp_center_px_preview.png", prev_px)
    stats = {
        "unit_equi": "rad",
        "unit_center": "px",
        "preview_range_equi": rng,
        "preview_range_px": rng_px,
        "disp_px_median": float(np.median(disp_px[valid_px])) if valid_px.any() else 0.0,
        "disp_px_p95": float(np.percentile(disp_px[valid_px], 95)) if valid_px.any() else 0.0,
        "disp_rad_max": float(disp_equi.max()),
        "valid_fraction": float(valid_px.mean()),
        "focal_px": f,
    }
    imageio.write_json(d / "disp_info.json", stats)
    ctx.log(
        f"disparity 중앙값 {stats['disp_px_median']:.1f}px, p95 {stats['disp_px_p95']:.1f}px, 유효 {stats['valid_fraction'] * 100:.0f}%"
    )
    return {"model": info.as_dict(), **{k: v for k, v in stats.items() if not isinstance(v, dict)}}
