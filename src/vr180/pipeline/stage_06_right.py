"""Stage 6 — Right eye 주변부 합성: pano_L 을 disp_full 로 전방 워핑, hole 채움, 원본 영역은 R_equi."""

from __future__ import annotations

import cv2
import numpy as np

from vr180.geometry import sphere, warp
from vr180.util import imageio

INDEX = 6
NAME = "right"
DIRNAME = "06_right"
PARAMS_KEY = "stage_06_right"
USES_MODELS = True


def inputs(job):
    return [
        job.path(3, "pano_L.png"),
        job.path(5, "disp_full.npy"),
        job.path(2, "R_equi.png"),
        job.path(2, "valid_mask.png"),
    ]


def outputs(job):
    d = job.stage_dir(INDEX)
    return [d / "pano_R.png", d / "hole_mask.png", d / "anaglyph.png", d / "warped_raw.png"]


def _fill_holes(rgb: np.ndarray, holes: np.ndarray, hole_px: int, inpainter, params: dict, ctx) -> np.ndarray:
    """작은 구멍은 Telea, 큰 구멍은 inpaint 모델 (있으면)."""
    if holes.max() == 0:
        return rgb
    n, labels, stats, _ = cv2.connectedComponentsWithStats((holes > 0).astype(np.uint8), connectivity=8)
    small = np.zeros_like(holes)
    big = np.zeros_like(holes)
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        (big if area > hole_px else small)[labels == k] = 255
    out = rgb
    if small.max() > 0:
        out = cv2.inpaint(out, small, 3, cv2.INPAINT_TELEA)
    if big.max() > 0:
        ctx.log(f"큰 구멍 {int((big > 0).sum())}px → {inpainter.info().name}")
        if inpainter.info().name.startswith("opencv"):
            out = inpainter.inpaint(out, big)
        else:
            # 큰 구멍은 구멍을 포함하는 영역을 타일로 잘라 모델에 넣는다 (equirect 왜곡은 구멍이 작아 무시)
            ys, xs = np.where(big > 0)
            y0, y1 = max(0, ys.min() - 128), min(rgb.shape[0], ys.max() + 128)
            x0, x1 = max(0, xs.min() - 128), min(rgb.shape[1], xs.max() + 128)
            crop = out[y0:y1, x0:x1]
            mcrop = cv2.dilate(big[y0:y1, x0:x1], np.ones((9, 9), np.uint8))
            res = int(params.get("tile_res", 1024))
            ch, cw = crop.shape[:2]
            scale = min(1.0, res / max(ch, cw))
            tw, th = max(64, int(cw * scale) // 8 * 8), max(64, int(ch * scale) // 8 * 8)
            crop_s = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
            m_s = cv2.resize(mcrop, (tw, th), interpolation=cv2.INTER_NEAREST)
            filled = inpainter.inpaint(
                crop_s,
                m_s,
                prompt=str(params.get("prompt", "")),
                negative_prompt=str(params.get("negative_prompt", "")),
                seed=int(params.get("seed", 0)),
                steps=int(params.get("steps", 30)),
                guidance=float(params.get("guidance", 7.0)),
            )
            filled = cv2.resize(filled, (cw, ch), interpolation=cv2.INTER_CUBIC)
            sel = big[y0:y1, x0:x1] > 0
            out = out.copy()
            out[y0:y1, x0:x1][sel] = filled[sel]
    return out


def run(job, params: dict, ctx) -> dict:
    from vr180.models.registry import get_inpainter

    pano_l = imageio.read_rgba(job.path(3, "pano_L.png"))[..., :3]
    disp = imageio.read_npy(job.path(5, "disp_full.npy"))
    r_equi = imageio.read_rgba(job.path(2, "R_equi.png"))
    valid = imageio.read_mask(job.path(2, "valid_mask.png")) > 127
    size = pano_l.shape[0]
    deg_px = sphere.equi_deg_per_pixel(size)
    d = job.stage_dir(INDEX)

    ctx.progress(0.1, "전방 워핑")
    warped, covered = warp.forward_warp(pano_l, disp, taps=int(params.get("splat_taps", 2)))
    holes = warp.hole_mask(covered)
    imageio.write_image(d / "warped_raw.png", warped)
    imageio.write_image(d / "hole_mask.png", holes)
    ctx.log(f"hole {float((holes > 0).mean()) * 100:.2f}%")

    ctx.progress(0.4, "hole 채우기")
    p3 = job.params.get("stage_03_outpaint", {})
    fill_params = {**p3, **params}
    inpainter = get_inpainter(params.get("inpainter", "opencv"), job.model_params())
    with inpainter:
        filled = _fill_holes(warped, holes, int(params.get("hole_px", 400)), inpainter, fill_params, ctx)

    # 원본 영역은 R_equi, 경계는 블렌딩
    ctx.progress(0.8, "합성")
    dist_in = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    blend_px = float(params.get("blend_deg", 8.0)) / deg_px
    w = np.clip(dist_in / max(blend_px, 1.0), 0, 1).astype(np.float32)
    w[~valid] = 0
    r_rgb = r_equi[..., :3].astype(np.float32)
    pano_r = (w[..., None] * r_rgb + (1 - w[..., None]) * filled.astype(np.float32)).astype(np.uint8)
    pano_r_rgba = np.dstack([pano_r, np.full((size, size), 255, np.uint8)])
    imageio.write_image(d / "pano_R.png", pano_r_rgba)

    ana = np.dstack([pano_l[..., 0], pano_r[..., 1], pano_r[..., 2]])
    imageio.write_image(d / "anaglyph.png", ana)
    return {"model": inpainter.info().as_dict(), "hole_fraction": float((holes > 0).mean())}
