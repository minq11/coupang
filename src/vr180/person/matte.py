"""경계 다듬기 — 사람 마스크를 머리카락까지 살린 알파로.

세 구역 규칙
  안쪽 (마스크를 core_px 만큼 깎은 곳)    → 무조건 불투명. 몸·들고 있는 물건이 반투명해지는 것 방지
  바깥 (마스크를 band_px 만큼 넓힌 밖)    → 무조건 투명. 옆 물체·다른 배경이 섞이는 것 방지
  경계 띠 (그 사이)                       → 매팅 모델(BiRefNet) 값. 머리카락·옷깃의 반투명
매팅 모델은 사람 덩어리마다 박스를 여유 있게 잘라 넣는다 (작게 나온 사람도 1024 해상도로 보게).
깜빡임 줄이기는 경계 띠에서만 앞·뒤 프레임 평균 (temporal_smooth).
"""

from __future__ import annotations

import cv2
import numpy as np

from vr180.person.models import Matter


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def zones(mask: np.ndarray, core_px: int, band_px: int) -> tuple[np.ndarray, np.ndarray]:
    """(core, outer) bool. 경계 띠 = outer & ~core."""
    # 원형 커널 erode/dilate 와 같은 결과를 거리 변환으로 (반지름 31px 에서 79ms → 수 ms, 1080x1920).
    # 침식: 가장 가까운 바깥 픽셀까지 거리 > r.  팽창: 가장 가까운 사람 픽셀까지 거리 ≤ r.
    m = mask.astype(np.uint8)
    if core_px > 0:
        core = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) > core_px
    else:
        core = mask.astype(bool)
    if band_px > 0 and m.any():
        outer = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= band_px
    else:
        outer = mask.astype(bool)
    return core, outer


def refine_alpha(
    rgb: np.ndarray,
    mask: np.ndarray,
    matter: Matter | None,
    *,
    core_px: int,
    band_px: int,
    crop_margin: float = 0.15,
) -> np.ndarray:
    """float32 알파 0..1 (H×W)."""
    mask = mask.astype(bool)
    h, w = mask.shape
    if not mask.any():
        return np.zeros((h, w), np.float32)
    core, outer = zones(mask, core_px, band_px)
    if matter is None:
        # 모델 없이: 마스크 경계를 띠 폭만큼 부드럽게
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(1.0, band_px / 3.0))
    else:
        soft = np.zeros((h, w), np.float32)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(outer.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            x, y, bw, bh = stats[k, :4]
            mx, my = int(bw * crop_margin), int(bh * crop_margin)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
            a = matter.alpha(rgb[y0:y1, x0:x1])
            sel = labels[y0:y1, x0:x1] == k
            region = soft[y0:y1, x0:x1]
            region[sel] = np.maximum(region[sel], a[sel])
    alpha = np.where(core, 1.0, np.where(outer, soft, 0.0)).astype(np.float32)
    return alpha


def band_of(mask: np.ndarray, core_px: int, band_px: int) -> np.ndarray:
    core, outer = zones(mask.astype(bool), core_px, band_px)
    return outer & ~core


def temporal_smooth(
    prev: np.ndarray | None, cur: np.ndarray, nxt: np.ndarray | None, band: np.ndarray
) -> np.ndarray:
    """경계 띠에서만 앞·뒤 프레임과 평균. 알파는 uint8 또는 float 모두 가능 (같은 dtype 으로 반환)."""
    stack = [x for x in (prev, cur, nxt) if x is not None]
    if len(stack) == 1 or not band.any():
        return cur
    mean = np.mean([x.astype(np.float32) for x in stack], axis=0)
    out = cur.astype(np.float32).copy()
    out[band] = mean[band]
    return out.astype(cur.dtype) if cur.dtype != np.float32 else out
