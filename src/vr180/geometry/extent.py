"""생성 범위(extent) 와 페이드 가중치. Stage 2 가 쓰고, Stage 3/5/6 이 입력으로 받는다.

규약
----
- 범위 정의는 캔버스 위의 **각도 기반 타원**. equirect 캔버스 좌표 (θ, φ) 에 대해
    r = sqrt((θ / (h/2))² + (φ / (v/2))²)
  r ≤ 1 이 "범위 안". θ, φ 는 도(°) 로 간주.
- 페이드: r ≤ 1 − f 에서 w = 1, r = 1 에서 w = 0, 사이는 smoothstep (3t² − 2t³).
  f 는 반지름 비율. extent_fade_deg → min(h/2, v/2) 로 나눠 환산.
- 모양 옵션: 'ellipse' (기본), 'rounded_rect' (사각형 + 모서리 radius).

API
---
- compute_extent_hv(hfov_deg, vfov_deg, params): 실제 사용할 (h_deg, v_deg) 를 결정.
  margin 규칙 (원본 화각 + 여유) 또는 고정 extent_h/v_deg.
- extent_weight(size, h_deg, v_deg, fade_deg, shape): (mask: bool S×S, weight: float32 S×S).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vr180.geometry import sphere


def compute_extent_hv(hfov_deg: float, vfov_deg: float, params: dict[str, Any]) -> tuple[float, float]:
    """실제 사용할 extent 각도 (h_deg, v_deg) 를 돌려준다.

    - extent_h_deg / extent_v_deg 가 명시(≠null)되면 그 값을 쓴다.
    - 아니면 'hfov + 2·extent_margin_h_deg' 과 같은 margin 규칙으로 계산.
    - 결과는 [0, 180] 로 클램프.
    """
    h = params.get("extent_h_deg")
    v = params.get("extent_v_deg")
    if h is None:
        margin_h = float(params.get("extent_margin_h_deg", 20.0))
        h = float(hfov_deg) + 2.0 * margin_h
    if v is None:
        margin_v = float(params.get("extent_margin_v_deg", 25.0))
        v = float(vfov_deg) + 2.0 * margin_v
    return float(min(max(h, 0.0), 180.0)), float(min(max(v, 0.0), 180.0))


def _ellipse_r(theta_deg: np.ndarray, phi_deg: np.ndarray, h_deg: float, v_deg: float) -> np.ndarray:
    """r = sqrt((θ/(h/2))² + (φ/(v/2))²). 입력 각도는 degree."""
    ah = max(h_deg / 2.0, 1e-6)
    av = max(v_deg / 2.0, 1e-6)
    tn = theta_deg / ah
    pn = phi_deg / av
    return np.sqrt(tn * tn + pn * pn).astype(np.float32)


def _rounded_rect_r(theta_deg: np.ndarray, phi_deg: np.ndarray, h_deg: float, v_deg: float) -> np.ndarray:
    """사각형 + 모서리 radius. 아직은 ellipse 와 거의 같게; 가장자리 블렌드만 완만하게."""
    ah = max(h_deg / 2.0, 1e-6)
    av = max(v_deg / 2.0, 1e-6)
    tn = np.abs(theta_deg) / ah
    pn = np.abs(phi_deg) / av
    # 모서리 반경 = 짧은 축의 20%
    k = 0.8
    inner = np.maximum(tn - k, 0.0) ** 2 + np.maximum(pn - k, 0.0) ** 2
    core = np.maximum(tn, pn)
    r = np.where(core > k, k + np.sqrt(inner), core)
    return r.astype(np.float32)


def extent_weight(
    size: int,
    h_deg: float,
    v_deg: float,
    fade_deg: float = 10.0,
    shape: str = "ellipse",
) -> tuple[np.ndarray, np.ndarray]:
    """캔버스 S×S 에서 (mask, weight).

    mask: bool. r ≤ 1 (페이드 띠 포함, 즉 weight > 0 영역).
    weight: float32. r ≤ 1 − f 에서 1, r = 1 에서 0, 사이는 smoothstep.
    둘 다 180 이면 '기능 꺼짐' → 전체 True + weight 1.
    """
    if is_disabled(h_deg, v_deg):
        return np.ones((size, size), dtype=bool), np.ones((size, size), dtype=np.float32)
    theta, phi = sphere.equi_angle_grid(size)  # rad
    theta_deg = np.degrees(theta).astype(np.float32)
    phi_deg = np.degrees(phi).astype(np.float32)
    if shape == "rounded_rect":
        r = _rounded_rect_r(theta_deg, phi_deg, h_deg, v_deg)
    else:
        r = _ellipse_r(theta_deg, phi_deg, h_deg, v_deg)
    # 페이드 비율. 안전하게 (0, 1) 로 클램프.
    f = float(fade_deg) / max(min(h_deg / 2.0, v_deg / 2.0), 1e-6)
    f = float(min(max(f, 0.0), 0.999))
    inner = 1.0 - f
    denom = max(1.0 - inner, 1e-6)
    t = np.clip((1.0 - r) / denom, 0.0, 1.0)
    w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    mask = (r <= 1.0).astype(bool)
    return mask, w


def is_disabled(h_deg: float, v_deg: float) -> bool:
    """둘 다 180 (혹은 그 이상) 이면 기능 꺼짐으로 간주 (기존 180×180 과 동일 동작)."""
    return h_deg >= 179.999 and v_deg >= 179.999
