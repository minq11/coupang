"""Stage 5 — 주변부 mono depth 와 스케일 정합 → 180° 전체 각도 disparity (disp_full)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vr180.geometry import reproject, sphere
from vr180.util import imageio
from vr180.util.fill import nearest_fill

INDEX = 5
NAME = "depth"
DIRNAME = "05_depth"
PARAMS_KEY = "stage_05_depth"
USES_MODELS = True


def inputs(job):
    return [job.path(3, "pano_L.png"), job.path(4, "disp_center_equi.npy"), job.path(2, "valid_mask.png")]


def outputs(job):
    d = job.stage_dir(INDEX)
    return [
        d / "mono_invdepth.npy",
        d / "disp_full.npy",
        d / "fit_report.json",
        d / "disp_full_preview.png",
        d / "mono_preview.png",
        d / "fit_scatter.png",
    ]


# ----------------------------------------------------------------------------- 피팅
def ransac_line(
    x: np.ndarray, y: np.ndarray, iters: int, thresh: float, seed: int = 0
) -> tuple[float, float, float]:
    """y ≈ a·x + b. 반환 (a, b, inlier_ratio). 마지막에 inlier 로 최소제곱 재피팅."""
    n = x.size
    if n < 2:
        return 1.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    best = (1.0, 0.0, -1)
    for _ in range(int(iters)):
        i, j = rng.integers(0, n, 2)
        if x[i] == x[j]:
            continue
        a = (y[j] - y[i]) / (x[j] - x[i])
        b = y[i] - a * x[i]
        cnt = int(np.count_nonzero(np.abs(y - (a * x + b)) < thresh))
        if cnt > best[2]:
            best = (a, b, cnt)
    a, b, cnt = best
    inl = np.abs(y - (a * x + b)) < thresh
    if inl.sum() >= 2:
        A = np.stack([x[inl], np.ones(inl.sum())], axis=1)
        sol, *_ = np.linalg.lstsq(A, y[inl], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        inl = np.abs(y - (a * x + b)) < thresh
    return float(a), float(b), float(inl.mean())


def affine_align(src: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """src 를 ref 에 맞추는 (a, b): ref ≈ a·src + b. 중앙값 기반 강건 추정 후 최소제곱."""
    if src.size < 10:
        return 1.0, 0.0
    s_med, r_med = np.median(src), np.median(ref)
    s_mad = np.median(np.abs(src - s_med)) + 1e-9
    r_mad = np.median(np.abs(ref - r_med)) + 1e-9
    a = r_mad / s_mad
    b = r_med - a * s_med
    res = np.abs(ref - (a * src + b))
    inl = res < 2.5 * np.median(res) + 1e-9
    if inl.sum() > 10:
        A = np.stack([src[inl], np.ones(inl.sum())], axis=1)
        sol, *_ = np.linalg.lstsq(A, ref[inl], rcond=None)
        a, b = float(sol[0]), float(sol[1])
    return float(a), float(b)


def _scatter_png(x, y, a, b, path, title: str):
    """matplotlib 없이 OpenCV 로 산점도 + 피팅선."""
    W = 640
    img = np.full((W, W, 3), 255, np.uint8)
    if x.size:
        xlo, xhi = np.percentile(x, [0.5, 99.5])
        ylo, yhi = np.percentile(y, [0.5, 99.5])
        xhi = xhi if xhi > xlo else xlo + 1e-6
        yhi = yhi if yhi > ylo else ylo + 1e-6
        px = ((x - xlo) / (xhi - xlo) * (W - 80) + 40).astype(int)
        py = (W - 40 - (y - ylo) / (yhi - ylo) * (W - 80)).astype(int)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < W)
        img[py[ok], px[ok]] = (60, 60, 200)
        for xx in (xlo, xhi):
            yy = a * xx + b
            cv2.circle(
                img,
                (
                    int((xx - xlo) / (xhi - xlo) * (W - 80) + 40),
                    int(W - 40 - (yy - ylo) / (yhi - ylo) * (W - 80)),
                ),
                3,
                (0, 0, 0),
                -1,
            )
        p0 = (40, int(W - 40 - (a * xlo + b - ylo) / (yhi - ylo) * (W - 80)))
        p1 = (W - 40, int(W - 40 - (a * xhi + b - ylo) / (yhi - ylo) * (W - 80)))
        cv2.line(img, p0, p1, (200, 30, 30), 2)
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        "x: mono inv-depth   y: stereo disp (rad)",
        (10, W - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    imageio.write_image(path, img)


# ----------------------------------------------------------------------------- mono depth
def _mono_tiled(pano_rgb: np.ndarray, est, params: dict, ctx) -> np.ndarray:
    size = pano_rgb.shape[0]
    tile_res = int(params.get("tile_res", 1024))
    tiles = reproject.make_tile_grid(
        params.get("tile_yaws_deg", [-60, 0, 60]),
        params.get("tile_pitches_deg", [-45, 0, 45]),
        float(params.get("tile_fov_deg", 90.0)),
        tile_res,
        skip_center=False,
    )
    pano_rgba = np.dstack([pano_rgb, np.full(pano_rgb.shape[:2], 255, np.uint8)])
    canvas = np.zeros((size, size), np.float32)
    wsum = np.zeros((size, size), np.float32)
    feather = int(params.get("feather_px", 96))
    n = len(tiles)
    for i, t in enumerate(tiles):
        ctx.progress(0.1 + 0.5 * i / n, f"mono depth 타일 {t.name}")
        rgb, _ = t.render_from_equi(pano_rgba)
        inv = est.inverse_depth(rgb)
        inv_equi, inside = t.splat_float_to_equi(inv, size)
        _, w = t.feather_weight(size, feather)
        overlap = inside & (wsum > 0.05)
        if overlap.sum() > 500:
            ref = canvas[overlap] / wsum[overlap]
            a, b = affine_align(inv_equi[overlap], ref)
            inv_equi = a * inv_equi + b
            ctx.log(f"  {t.name}: 정렬 a={a:.3f} b={b:.4f}")
        canvas += inv_equi * w
        wsum += w
        ctx.check_cancelled()
    out = np.where(wsum > 0, canvas / np.maximum(wsum, 1e-6), 0).astype(np.float32)
    if (wsum <= 0).any():
        out = nearest_fill(out, wsum > 0)
    return out


def _mono_whole(pano_rgb: np.ndarray, est, ctx) -> np.ndarray:
    ctx.progress(0.2, "mono depth (전체)")
    size = pano_rgb.shape[0]
    small = cv2.resize(pano_rgb, (min(size, 1536), min(size, 1536)), interpolation=cv2.INTER_AREA)
    inv = est.inverse_depth(small)
    return cv2.resize(inv, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def run(job, params: dict, ctx) -> dict:
    from vr180.models.registry import get_depth_estimator

    pano = imageio.read_rgba(job.path(3, "pano_L.png"))
    pano_rgb = pano[..., :3]
    disp_c = imageio.read_npy(job.path(4, "disp_center_equi.npy"))
    valid = imageio.read_mask(job.path(2, "valid_mask.png")) > 127
    size = pano.shape[0]
    deg_px = sphere.equi_deg_per_pixel(size)
    d = job.stage_dir(INDEX)

    # Stage 3 과 같은 타일 분할을 쓰기 위해 그 파라미터를 읽는다 (데이터만, 함수는 아님)
    p3 = job.params.get("stage_03_outpaint", {})
    tile_params = {
        k: p3.get(k, v)
        for k, v in {
            "tile_res": 1024,
            "tile_fov_deg": 90.0,
            "tile_yaws_deg": [-60, 0, 60],
            "tile_pitches_deg": [-45, 0, 45],
            "feather_px": 64,
        }.items()
    }

    est = get_depth_estimator(params.get("estimator", "da2_large"), job.model_params())
    model_info = {"name": "none"}
    if est is None:
        ctx.log("estimator=none: stereo disparity 를 바깥으로 최근접 외삽 (뼈대용)")
        mono = nearest_fill(disp_c, valid)
        mono = cv2.GaussianBlur(mono, (0, 0), size / 200.0)
    else:
        model_info = est.info().as_dict()
        ctx.log(f"mono depth = {model_info['name']}")
        with est:
            mono = (
                _mono_tiled(pano_rgb, est, tile_params, ctx)
                if params.get("tile_mode", True)
                else _mono_whole(pano_rgb, est, ctx)
            )
    imageio.write_npy(d / "mono_invdepth.npy", mono)
    mprev, mrng = imageio.colormap_preview(mono)
    imageio.write_image(d / "mono_preview.png", mprev)

    # 경계 안쪽 띠에서 선형 피팅
    ctx.progress(0.7, "스케일 정합")
    dist_in = cv2.distanceTransform(
        valid.astype(np.uint8), cv2.DIST_L2, 5
    )  # 유효 영역 안에서 경계까지 거리(px)
    band_px = float(params.get("band_deg", 5.0)) / deg_px
    blend_px = float(params.get("blend_deg", 8.0)) / deg_px
    band = valid & (dist_in <= band_px) & (dist_in > 0)
    xs = mono[band].astype(np.float64)
    ys = disp_c[band].astype(np.float64)
    rng = np.random.default_rng(0)
    if xs.size > 20000:
        idx = rng.choice(xs.size, 20000, replace=False)
        xs, ys = xs[idx], ys[idx]
    a, b, inl = ransac_line(
        xs, ys, int(params.get("ransac_iters", 500)), float(params.get("ransac_thresh", 0.004))
    )
    fit_src = "ransac"
    if params.get("override_a") is not None:
        a = float(params["override_a"])
        fit_src = "override"
    if params.get("override_b") is not None:
        b = float(params["override_b"])
        fit_src = "override"
    if est is None:
        a, b, inl, fit_src = 1.0, 0.0, 1.0, "identity(estimator=none)"
    _scatter_png(
        xs, ys, a, b, d / "fit_scatter.png", f"a={a:.4g} b={b:.4g} inlier={inl * 100:.0f}% ({fit_src})"
    )
    warn = ""
    if inl < float(params.get("min_inlier_ratio", 0.6)):
        warn = f"inlier 비율 {inl * 100:.0f}% < {params.get('min_inlier_ratio', 0.6) * 100:.0f}% — band_deg 조정 권장"
        ctx.log("경고: " + warn)

    disp_mono = (a * mono + b).astype(np.float32)
    w_in = np.clip(dist_in / max(blend_px, 1.0), 0, 1).astype(np.float32)
    w_in[~valid] = 0
    disp_full = w_in * disp_c + (1 - w_in) * disp_mono

    # 극지방 완화
    _, phi = sphere.equi_angle_grid(size)
    pstart = math.radians(float(params.get("polar_start_deg", 75.0)))
    pmin = float(params.get("polar_min_disp", 0.0))
    t = np.clip((np.abs(phi) - pstart) / max(math.pi / 2 - pstart, 1e-6), 0, 1).astype(np.float32)
    t = t * t * (3 - 2 * t)  # smoothstep
    disp_full = disp_full * (1 - t) + np.minimum(disp_full, pmin) * t
    disp_full = np.clip(disp_full, 0, None).astype(np.float32)

    imageio.write_npy(d / "disp_full.npy", disp_full)
    prev, rng_full = imageio.colormap_preview(disp_full)
    imageio.write_image(d / "disp_full_preview.png", prev)
    report = {
        "a": a,
        "b": b,
        "inlier_ratio": inl,
        "fit_source": fit_src,
        "num_band_samples": int(xs.size),
        "band_deg": params.get("band_deg", 5.0),
        "blend_deg": params.get("blend_deg", 8.0),
        "unit": "rad",
        "preview_range_full": rng_full,
        "preview_range_mono": mrng,
        "disp_full_max": float(disp_full.max()),
        "warning": warn,
        "model": model_info,
    }
    imageio.write_json(d / "fit_report.json", report)
    ctx.log(f"피팅 a={a:.4g} b={b:.4g} inlier={inl * 100:.0f}% disp_full max={disp_full.max():.4f}rad")
    return {"model": model_info, "a": a, "b": b, "inlier_ratio": inl}
