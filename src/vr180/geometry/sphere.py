"""구면 좌표 규약 — 이 파일에서만 정의한다. 다른 곳에서 재구현 금지.

규약
----
- 카메라 forward = +Z, right = +X, up = +Y (오른손 좌표계가 아니라 '보는 사람' 기준 좌표).
- yaw θ: +Z 에서 +X 쪽으로 양수 (오른쪽). pitch φ: +Y 쪽으로 양수 (위). 둘 다 라디안.
- 방향 벡터 d = (sinθ·cosφ, sinφ, cosθ·cosφ).
- half-equirect 캔버스 S×S: 픽셀 (u, v) 의 중심이 각도 (θ, φ) 에 대응.
    θ = (u + 0.5) / S · π − π/2   ∈ (−π/2, π/2)      왼쪽 열 = −90°
    φ = π/2 − (v + 0.5) / S · π   ∈ (−π/2, π/2)      위쪽 행 = +90°
- 핀홀 이미지 W×H, 수평 FOV hfov: f = (W/2) / tan(hfov/2), 주점 = (W/2, H/2).
    px = cx + f · (d.x / d.z),  py = cy − f · (d.y / d.z)   (d.z > 0 일 때만 유효)
"""

from __future__ import annotations

import math

import numpy as np

HALF_PI = math.pi / 2.0


# ----------------------------------------------------------------------------- 각도 <-> 벡터
def angles_to_dir(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """(θ, φ) 라디안 → 단위 방향 벡터 (..., 3)."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    cp = np.cos(phi)
    return np.stack([np.sin(theta) * cp, np.sin(phi), np.cos(theta) * cp], axis=-1)


def dir_to_angles(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """단위 방향 벡터 (..., 3) → (θ, φ). 정규화는 호출자가 보장하지 않아도 된다."""
    d = np.asarray(d, dtype=np.float64)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    d = d / n
    theta = np.arctan2(d[..., 0], d[..., 2])
    phi = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    return theta, phi


# ----------------------------------------------------------------------------- equirect 픽셀 <-> 각도
def equi_pixel_to_angles(u: np.ndarray, v: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """half-equirect 픽셀 좌표 (연속값, 픽셀 중심이 정수+0) → (θ, φ)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    theta = (u + 0.5) / size * math.pi - HALF_PI
    phi = HALF_PI - (v + 0.5) / size * math.pi
    return theta, phi


def angles_to_equi_pixel(theta: np.ndarray, phi: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """(θ, φ) → half-equirect 픽셀 좌표 (연속값)."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    u = (theta + HALF_PI) / math.pi * size - 0.5
    v = (HALF_PI - phi) / math.pi * size - 0.5
    return u, v


def equi_angle_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """S×S 캔버스 전체의 (θ, φ) 격자 (각각 S×S, float64)."""
    u = np.arange(size, dtype=np.float64)
    v = np.arange(size, dtype=np.float64)
    theta_row = (u + 0.5) / size * math.pi - HALF_PI
    phi_col = HALF_PI - (v + 0.5) / size * math.pi
    theta = np.broadcast_to(theta_row[None, :], (size, size))
    phi = np.broadcast_to(phi_col[:, None], (size, size))
    return theta, phi


def equi_deg_per_pixel(size: int) -> float:
    """half-equirect 에서 픽셀 1개가 차지하는 각도 (θ, φ 동일)."""
    return 180.0 / size


# ----------------------------------------------------------------------------- 핀홀 카메라
def focal_px(width: int, hfov_rad: float) -> float:
    return (width / 2.0) / math.tan(hfov_rad / 2.0)


def vfov_from_hfov(hfov_rad: float, width: int, height: int) -> float:
    return 2.0 * math.atan(math.tan(hfov_rad / 2.0) * height / width)


def dir_to_pinhole(
    d: np.ndarray, width: int, height: int, hfov_rad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """방향 벡터 → 핀홀 이미지 픽셀 좌표 (px, py) 와 유효 마스크 (d.z > 0 이고 이미지 안)."""
    d = np.asarray(d, dtype=np.float64)
    f = focal_px(width, hfov_rad)
    cx, cy = width / 2.0, height / 2.0
    z = d[..., 2]
    front = z > 1e-9
    zs = np.where(front, z, 1.0)
    x = d[..., 0] / zs
    y = d[..., 1] / zs
    px = cx + f * x
    py = cy - f * y
    # 픽셀 중심 규약: 픽셀 i 의 중심은 i+0.5 → 이미지 안쪽은 [0, W) x [0, H)
    inside = front & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px - 0.5  # cv2.remap 은 픽셀 중심을 정수로 본다
    py = py - 0.5
    return px, py, inside


def pinhole_to_dir(px: np.ndarray, py: np.ndarray, width: int, height: int, hfov_rad: float) -> np.ndarray:
    """핀홀 픽셀 좌표 (정수 = 픽셀 중심 규약, 즉 dir_to_pinhole 의 역) → 단위 방향 벡터."""
    px = np.asarray(px, dtype=np.float64) + 0.5
    py = np.asarray(py, dtype=np.float64) + 0.5
    f = focal_px(width, hfov_rad)
    cx, cy = width / 2.0, height / 2.0
    x = (px - cx) / f
    y = (cy - py) / f
    d = np.stack([x, y, np.ones_like(x)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d


# ----------------------------------------------------------------------------- 회전
def rotation_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """카메라를 yaw(θ) 만큼 돌리고 pitch(φ) 만큼 올린 자세. R @ d_cam = d_world.

    카메라 로컬 forward (0,0,1) 이 world 에서 angles_to_dir(yaw, pitch) 를 가리키도록 한다.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # pitch: X 축 회전 (Y↔Z), +pitch 가 forward 를 +Y 로 올린다
    r_pitch = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]], dtype=np.float64)
    # yaw: Y 축 회전 (Z↔X), +yaw 가 forward 를 +X 로 돌린다
    r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    return r_yaw @ r_pitch


def world_to_cam(d_world: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return np.asarray(d_world) @ rot  # (R^T d) 를 행벡터로


def cam_to_world(d_cam: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return np.asarray(d_cam) @ rot.T


# ----------------------------------------------------------------------------- disparity 단위
def pixel_disparity_to_angular(
    d_px: np.ndarray, x_norm: np.ndarray, f: float, exact: bool = True
) -> np.ndarray:
    """핀홀 픽셀 시차 → 각도 시차 (rad).

    d_px = f·b/Z. 점이 정규화 좌표 x = X/Z 에 있을 때 θ = atan(x) 이므로
    Δθ = dθ/dx · Δx = (b/Z) / (1 + x²) = d_px / f / (1 + x²).
    exact=False 면 설계서의 근사식 d_px / f.
    """
    d_px = np.asarray(d_px, dtype=np.float32)
    if not exact:
        return d_px / np.float32(f)
    x_norm = np.asarray(x_norm, dtype=np.float32)
    return d_px / np.float32(f) / (1.0 + x_norm * x_norm)
