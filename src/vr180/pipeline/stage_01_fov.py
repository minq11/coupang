"""Stage 1 — 원본 화각 추정. 기본은 수동 hfov_deg, 옵션으로 GeoCalib 자동 추정."""

from __future__ import annotations

import math

from vr180.geometry import sphere
from vr180.util import imageio

INDEX = 1
NAME = "fov"
DIRNAME = "01_fov"
PARAMS_KEY = "stage_01_fov"
USES_MODELS = True


def inputs(job):
    return [job.path(0, "L.png")]


def outputs(job):
    return [job.path(INDEX, "camera.json")]


def run(job, params: dict, ctx) -> dict:
    left = imageio.read_rgb(job.path(0, "L.png"))
    h, w = left.shape[:2]
    hfov = float(params.get("hfov_deg", 80.0))
    source = "manual"
    conf = 1.0
    est_hfov = None
    model_name = ""
    if params.get("auto_estimate", False):
        from vr180.models.registry import get_fov_estimator

        est = get_fov_estimator(params.get("estimator", "geocalib"), job.model_params())
        if est is not None:
            try:
                with est:
                    est_hfov, conf = est.estimate_hfov(left)
                    model_name = est.info().name
                ctx.log(f"자동 추정 hfov={est_hfov} conf={conf:.2f}")
            except Exception as e:  # 추정 실패해도 파이프라인은 진행
                ctx.log(f"FOV 자동 추정 실패 ({e}) — 수동값 {hfov}° 유지")
            if (
                est_hfov is not None
                and conf >= float(params.get("auto_min_confidence", 0.5))
                and 20 < est_hfov < 150
            ):
                hfov = float(est_hfov)
                source = "auto"
            else:
                source = "manual(auto 신뢰도 낮음)"
    hfov_rad = math.radians(hfov)
    vfov_rad = sphere.vfov_from_hfov(hfov_rad, w, h)
    cam = {
        "width": w,
        "height": h,
        "hfov_deg": hfov,
        "vfov_deg": math.degrees(vfov_rad),
        "focal_px": sphere.focal_px(w, hfov_rad),
        "source": source,
        "confidence": conf,
        "auto_hfov_deg": est_hfov,
        "estimator": model_name,
    }
    imageio.write_json(job.path(INDEX, "camera.json"), cam)
    ctx.log(f"hfov={hfov:.2f}° vfov={cam['vfov_deg']:.2f}° f={cam['focal_px']:.1f}px ({source})")
    return {"model": model_name, "hfov_deg": hfov}
