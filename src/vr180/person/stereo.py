"""좌우 영상 만들기와 포장.

깊이 → 시차 (픽셀)
  - 사람 픽셀의 inverse depth 를 1~99 백분위로 0..1 로 정규화 (영상은 전체 프레임 공통 값을 써서 깊이 출렁임 방지)
  - 사람 바깥 깊이는 가장 가까운 사람 깊이로 덮는다 → 워핑 때 가장자리가 늘어나지 않음
  - 초점(시차 0): 가장 앞 사람(front, 사람 깊이 85 백분위) 또는 평균(mean). 초점보다 가까우면 +, 멀면 −
  - relief_px: 가장 먼 사람 ~ 가장 가까운 사람의 전체 시차 폭. 2m 앞 사람 몸 두께는 1080 줄 기준 약 6px
양쪽 눈 (교차 시차): 가까운 픽셀은 왼눈에서 오른쪽(+d/2), 오른눈에서 왼쪽(−d/2)
  - 사람 바깥 구멍은 투명, 사람 안쪽 구멍(팔 뒤 등)만 Telea 로 채움
포장 (layout "rgba_sbs_v1", 플레이어와 약속)
  위 절반: [ 왼눈 컬러 | 오른눈 컬러 ]   아래 절반: [ 왼눈 알파 | 오른눈 알파 ] (회색)
  투명한 곳의 컬러는 사람 색을 번지게 채워 검은 테두리를 막는다.
  초록 배경 SBS: 기존 플레이어용 [ 왼눈 | 오른눈 ] 을 크로마키 초록 위에 합성.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from vr180.geometry.warp import splat_horizontal
from vr180.util.fill import nearest_fill, nearest_fill_rgb

LAYOUT = "rgba_sbs_v1"
GREEN = (0, 177, 64)


def depth_stats(inv_depth: np.ndarray, mask: np.ndarray, front_pct: float = 85.0) -> dict | None:
    m = mask.astype(bool)
    if m.sum() < 16:
        return None
    v = inv_depth[m].astype(np.float32)
    lo, hi = np.percentile(v, [1.0, 99.0])
    if hi - lo < 1e-6:
        return {"lo": float(lo), "hi": float(lo) + 1e-6, "t_front": 1.0, "t_mean": 1.0}
    t = np.clip((v - lo) / (hi - lo), 0, 1)
    return {
        "lo": float(lo),
        "hi": float(hi),
        "t_front": float(np.percentile(t, front_pct)),
        "t_mean": float(np.median(t)),
    }


def disparity_px(
    inv_depth: np.ndarray,
    mask: np.ndarray,
    *,
    lo: float,
    hi: float,
    t0: float,
    relief_px: float,
) -> np.ndarray:
    m = mask.astype(bool)
    h, w = m.shape
    if not m.any() or relief_px == 0:
        return np.zeros((h, w), np.float32)
    d = inv_depth.astype(np.float32)
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    t = nearest_fill(t, m)  # 사람 바깥 = 가장 가까운 사람 깊이
    return (float(relief_px) * (t - float(t0))).astype(np.float32)


def eye_views(rgb: np.ndarray, alpha_u8: np.ndarray, disp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """→ (왼눈 RGBA, 오른눈 RGBA) uint8."""
    rgba = np.dstack([rgb, alpha_u8]).astype(np.uint8)
    src = alpha_u8 > 0
    max_shift = int(math.ceil(float(np.abs(disp).max()) / 2.0)) if disp.size else 0
    out = []
    for sign in (+0.5, -0.5):  # 왼눈: 가까우면 오른쪽 / 오른눈: 가까우면 왼쪽
        if max_shift == 0:
            out.append(rgba.copy())
            continue
        eye, cov = splat_horizontal(rgba, sign * disp, disp, src_mask=src, taps=2)
        person = eye[..., 3] > 127
        if person.any():
            kw = 2 * max_shift + 3
            hull = cv2.morphologyEx(person.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, kw), np.uint8)) > 0
            inner = (hull & ~cov).astype(np.uint8) * 255
            if inner.any():
                sel = inner > 0
                c = cv2.inpaint(np.ascontiguousarray(eye[..., :3]), inner, 3, cv2.INPAINT_TELEA)
                a = cv2.inpaint(np.ascontiguousarray(eye[..., 3]), inner, 3, cv2.INPAINT_TELEA)
                eye[..., :3][sel] = c[sel]
                eye[..., 3][sel] = a[sel]
        out.append(eye)
    return out[0], out[1]


def bleed_color(rgba: np.ndarray, solid_thr: int = 128, clear_thr: int = 16) -> np.ndarray:
    """투명한 곳(알파 < clear_thr)의 컬러를 가까운 사람 색으로 채운다. 플레이어가 컬러·알파를 따로 샘플링해도
    경계에 검은 테두리가 안 생기게."""
    rgb = rgba[..., :3]
    a = rgba[..., 3]
    solid = a >= solid_thr
    if not solid.any():
        return rgb.copy()
    filled = nearest_fill_rgb(rgb, solid)
    filled = cv2.blur(filled, (5, 5))
    out = rgb.copy()
    clear = a < clear_thr
    out[clear] = filled[clear]
    return out


def pack(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """layout rgba_sbs_v1 → (2H, 2W, 3) uint8."""
    top = np.concatenate([bleed_color(left), bleed_color(right)], axis=1)
    al = np.repeat(left[..., 3:4], 3, axis=2)
    ar = np.repeat(right[..., 3:4], 3, axis=2)
    bottom = np.concatenate([al, ar], axis=1)
    return np.ascontiguousarray(np.concatenate([top, bottom], axis=0))


def green_sbs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    def comp(x):
        a = x[..., 3:4].astype(np.float32) / 255.0
        g = np.array(GREEN, np.float32)
        return (x[..., :3].astype(np.float32) * a + g * (1 - a)).astype(np.uint8)

    return np.ascontiguousarray(np.concatenate([comp(left), comp(right)], axis=1))


def preview(left: np.ndarray, right: np.ndarray, cell: int = 24) -> np.ndarray:
    """GUI 미리보기: [왼눈 체크무늬 합성 | 애너글리프]."""
    h, w = left.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where((((yy // cell) + (xx // cell)) % 2)[..., None] == 0, 70, 120).astype(np.float32)
    a = left[..., 3:4].astype(np.float32) / 255.0
    lc = (left[..., :3] * a + checker * (1 - a)).astype(np.uint8)
    gl = cv2.cvtColor(left[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) * (left[..., 3] / 255.0)
    gr = cv2.cvtColor(right[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) * (right[..., 3] / 255.0)
    ana = np.dstack([gl, gr, gr]).astype(np.uint8)
    return np.concatenate([lc, ana], axis=1)
