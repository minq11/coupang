"""disparity 기반 전방 워핑(splatting) 과 hole 마스크.

half-equirect 에서 시차는 수평 각도 이동이고, 위도 φ 에서 같은 깊이의 각도 이동은 cosφ 로 나눠진다:
    θ_R = θ_L − disp(θ, φ) / cosφ        (disp: 수평선 기준 각도 disparity, rad, 양수 = 가까움)
픽셀 단위로는 du = −disp / cosφ · (S / π).
"""

from __future__ import annotations

import math

import numpy as np

from vr180.geometry import sphere


def disparity_to_du(disp_rad: np.ndarray, size: int, min_cos: float = 0.05) -> np.ndarray:
    """각도 disparity (S×S, rad) → 수평 픽셀 이동량 du (float32). 극에서는 cosφ 를 min_cos 로 클램프."""
    _, phi = sphere.equi_angle_grid(size)
    cosphi = np.maximum(np.cos(phi), min_cos)
    return (-np.asarray(disp_rad, dtype=np.float64) / cosphi * (size / math.pi)).astype(np.float32)


def forward_warp(
    rgb: np.ndarray,
    disp_rad: np.ndarray,
    src_mask: np.ndarray | None = None,
    taps: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Left RGB 를 disparity 로 Right 위치에 splat.

    rgb: S×S×3 uint8, disp_rad: S×S float32, src_mask: 워핑할 픽셀 (None 이면 전체).
    반환: warped (S×S×3 uint8), covered (S×S bool; False = hole).
    z-buffer: 같은 목적지에 여러 소스가 오면 disparity 큰(가까운) 쪽이 이긴다.
    """
    size = rgb.shape[0]
    du = disparity_to_du(disp_rad, size)
    ys, xs = np.mgrid[0:size, 0:size]
    if src_mask is None:
        src_mask = np.ones((size, size), dtype=bool)
    sel = src_mask
    sy = ys[sel]
    sx = xs[sel]
    tx_f = sx + du[sel]
    dz = np.asarray(disp_rad, dtype=np.float32)[sel]
    color = rgb[sel]

    if taps == 1:
        tx_list = [np.round(tx_f).astype(np.int64)]
    else:
        tx_list = [np.floor(tx_f).astype(np.int64), np.floor(tx_f).astype(np.int64) + 1]

    zbuf = np.full((size, size), -np.inf, dtype=np.float32)
    out = np.zeros_like(rgb)
    covered = np.zeros((size, size), dtype=bool)

    for tx in tx_list:
        ok = (tx >= 0) & (tx < size)
        ty_i = sy[ok]
        tx_i = tx[ok]
        d_i = dz[ok]
        np.maximum.at(zbuf, (ty_i, tx_i), d_i)
    for tx in tx_list:
        ok = (tx >= 0) & (tx < size)
        ty_i = sy[ok]
        tx_i = tx[ok]
        d_i = dz[ok]
        win = d_i >= zbuf[ty_i, tx_i] - 1e-7
        out[ty_i[win], tx_i[win]] = color[ok][win]
        covered[ty_i[win], tx_i[win]] = True
    return out, covered


def forward_warp_float(
    arr: np.ndarray, disp_rad: np.ndarray, src_mask: np.ndarray | None = None, taps: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """스칼라 맵(예: disparity 자체)을 같은 규칙으로 워핑. 반환: warped float32, covered."""
    size = arr.shape[0]
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    # 색 대신 인덱스를 실어 보내서 값을 가져온다
    du = disparity_to_du(disp_rad, size)
    ys, xs = np.mgrid[0:size, 0:size]
    if src_mask is None:
        src_mask = np.ones((size, size), dtype=bool)
    sel = src_mask
    sy, sx = ys[sel], xs[sel]
    tx_f = sx + du[sel]
    dz = np.asarray(disp_rad, dtype=np.float32)[sel]
    vals = np.asarray(arr, dtype=np.float32)[sel]
    tx_list = (
        [np.round(tx_f).astype(np.int64)]
        if taps == 1
        else [np.floor(tx_f).astype(np.int64), np.floor(tx_f).astype(np.int64) + 1]
    )
    zbuf = np.full((size, size), -np.inf, dtype=np.float32)
    out = np.zeros((size, size), dtype=np.float32)
    covered = np.zeros((size, size), dtype=bool)
    for tx in tx_list:
        ok = (tx >= 0) & (tx < size)
        np.maximum.at(zbuf, (sy[ok], tx[ok]), dz[ok])
    for tx in tx_list:
        ok = (tx >= 0) & (tx < size)
        ty_i, tx_i, d_i = sy[ok], tx[ok], dz[ok]
        win = d_i >= zbuf[ty_i, tx_i] - 1e-7
        out[ty_i[win], tx_i[win]] = vals[ok][win]
        covered[ty_i[win], tx_i[win]] = True
    del rgb
    return out, covered


def hole_mask(covered: np.ndarray, region: np.ndarray | None = None) -> np.ndarray:
    """hole = 덮이지 않은 픽셀 (region 안에서만). uint8 255 = hole."""
    h = ~covered
    if region is not None:
        h &= region
    return h.astype(np.uint8) * 255
