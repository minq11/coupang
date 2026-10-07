"""Stage 0 — SBS 분리 + 정류(수직 시차) 검사."""

from __future__ import annotations

import cv2
import numpy as np

from vr180.util import imageio

INDEX = 0
NAME = "split"
DIRNAME = "00_split"
PARAMS_KEY = "stage_00_split"
USES_MODELS = False


def inputs(job):
    return [job.input_path]


def outputs(job):
    d = job.stage_dir(INDEX)
    return [d / "L.png", d / "R.png", d / "rectify_report.json"]


def _match_vertical_parallax(left: np.ndarray, right: np.ndarray, matcher: str, max_features: int) -> dict:
    """특징점 매칭으로 dy = y_R − y_L 통계. 반환 dict 는 rectify_report 에 들어간다."""
    h, w = left.shape[:2]
    scale = 1.0
    if w > 1920:
        scale = 1920.0 / w
        left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    pts_l = pts_r = None
    used = "sift"
    if matcher == "lightglue":
        try:
            pts_l, pts_r = _lightglue(left, right)
            used = "lightglue"
        except Exception:
            pts_l = pts_r = None
    if pts_l is None:
        sift = cv2.SIFT_create(nfeatures=int(max_features))
        k1, d1 = sift.detectAndCompute(gl, None)
        k2, d2 = sift.detectAndCompute(gr, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return {
                "matcher": used,
                "num_matches": 0,
                "dy_median_px": 0.0,
                "dy_mean_abs_px": 0.0,
                "dx_median_px": 0.0,
                "dy_samples": [],
            }
        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn = bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
        pts_l = np.float32([k1[m.queryIdx].pt for m in good])
        pts_r = np.float32([k2[m.trainIdx].pt for m in good])

    if len(pts_l) == 0:
        return {
            "matcher": used,
            "num_matches": 0,
            "dy_median_px": 0.0,
            "dy_mean_abs_px": 0.0,
            "dx_median_px": 0.0,
            "dy_samples": [],
        }
    d = (pts_r - pts_l) / scale
    dx, dy = d[:, 0], d[:, 1]
    # 스테레오 쌍이면 dx 는 음수(오른쪽 눈에서 왼쪽으로 이동)이고 |dy| 는 작아야 한다. 엉뚱한 매칭 제거.
    keep = (np.abs(dy) < 0.1 * h) & (np.abs(dx) < 0.5 * w)
    dx, dy = dx[keep], dy[keep]
    if dy.size == 0:
        return {
            "matcher": used,
            "num_matches": 0,
            "dy_median_px": 0.0,
            "dy_mean_abs_px": 0.0,
            "dx_median_px": 0.0,
            "dy_samples": [],
        }
    rng = np.random.default_rng(0)
    samp = dy if dy.size <= 2000 else rng.choice(dy, 2000, replace=False)
    return {
        "matcher": used,
        "num_matches": int(dy.size),
        "dy_median_px": float(np.median(dy)),
        "dy_mean_abs_px": float(np.mean(np.abs(dy))),
        "dy_p95_abs_px": float(np.percentile(np.abs(dy), 95)),
        "dx_median_px": float(np.median(dx)),
        "dx_min_px": float(np.min(dx)),
        "dx_max_px": float(np.max(dx)),
        "dy_samples": [round(float(v), 3) for v in samp],
    }


def _lightglue(left: np.ndarray, right: np.ndarray):
    import torch
    from lightglue import LightGlue, SuperPoint  # type: ignore
    from lightglue.utils import rbd  # type: ignore

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = SuperPoint(max_num_keypoints=2048).eval().to(dev)
    mat = LightGlue(features="superpoint").eval().to(dev)

    def to_t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()[None].to(dev) / 255.0

    f0 = ext.extract(to_t(left))
    f1 = ext.extract(to_t(right))
    m = mat({"image0": f0, "image1": f1})
    f0, f1, m = [rbd(x) for x in [f0, f1, m]]
    idx = m["matches"]
    p0 = f0["keypoints"][idx[:, 0]].cpu().numpy()
    p1 = f1["keypoints"][idx[:, 1]].cpu().numpy()
    return p0.astype(np.float32), p1.astype(np.float32)


def run(job, params: dict, ctx) -> dict:
    img = imageio.read_rgb(job.input_path)
    h, w = img.shape[:2]
    if w % 2 != 0:
        ctx.log(f"경고: 입력 폭 {w} 이 홀수 — 마지막 열을 버립니다")
        img = img[:, : w - 1]
        w -= 1
    half = w // 2
    left, right = img[:, :half].copy(), img[:, half:].copy()
    if params.get("swap_lr", False):
        left, right = right, left
        ctx.log("swap_lr: 좌우 순서를 뒤집음")
    ctx.progress(0.2, "특징점 매칭")
    rep = _match_vertical_parallax(
        left, right, params.get("matcher", "sift"), params.get("max_features", 4000)
    )
    rep["width"], rep["height"] = half, h
    rep["swap_lr"] = bool(params.get("swap_lr", False))
    warn = abs(rep["dy_median_px"]) > float(params.get("dy_warn_px", 1.0))
    rep["warning"] = (
        f"수직 시차 중앙값 {rep['dy_median_px']:.2f}px — 정류가 안 된 입력일 수 있음 (rectify 권장)"
        if warn
        else ""
    )
    if warn:
        ctx.log("경고: " + rep["warning"])
    rep["rectified"] = False
    rep["applied_shift_px"] = 0.0
    if params.get("rectify", False) and rep["num_matches"] > 0 and abs(rep["dy_median_px"]) > 0.05:
        shift = -rep["dy_median_px"]
        m = np.float32([[1, 0, 0], [0, 1, shift]])
        right = cv2.warpAffine(right, m, (half, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        rep["rectified"] = True
        rep["applied_shift_px"] = float(shift)
        ctx.log(f"rectify: R 을 y 방향 {shift:+.2f}px 이동")
    ctx.progress(0.8, "저장")
    d = job.stage_dir(INDEX)
    imageio.write_image(d / "L.png", left)
    imageio.write_image(d / "R.png", right)
    imageio.write_json(d / "rectify_report.json", rep)
    return {"num_matches": rep["num_matches"], "dy_median_px": rep["dy_median_px"]}
