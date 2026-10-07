"""합성 3D SBS 테스트 이미지 생성 — 실제 사진 없이 파이프라인을 끝까지 돌려 보기 위한 것.

바닥 체커보드 + 벽 + 떠 있는 상자 몇 개를 핀홀 카메라 두 대(baseline 만큼 X 이동)로 레이트레이싱한다.
"""

from __future__ import annotations

import math

import numpy as np


def _render(width: int, height: int, hfov_deg: float, eye_x: float, seed: int = 0) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return _render_impl(width, height, hfov_deg, eye_x, seed)


def _render_impl(width: int, height: int, hfov_deg: float, eye_x: float, seed: int = 0) -> np.ndarray:
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    ys, xs = np.mgrid[0:height, 0:width]
    dx = (xs + 0.5 - width / 2) / f
    dy = -(ys + 0.5 - height / 2) / f
    dz = np.ones_like(dx)
    n = np.sqrt(dx * dx + dy * dy + 1)
    dx, dy, dz = dx / n, dy / n, dz / n
    cam_y = 1.6  # 눈높이 (m)
    img = np.zeros((height, width, 3), np.float32)
    depth = np.full((height, width), np.inf, np.float32)

    # 바닥 y=0
    t = np.where(dy < -1e-6, -cam_y / np.where(dy < -1e-6, dy, -1), np.inf)
    px, pz = eye_x + dx * t, dz * t
    check = ((np.floor(px) + np.floor(pz)) % 2).astype(bool)
    floor_col = np.where(check[..., None], (0.75, 0.72, 0.65), (0.35, 0.33, 0.30))
    hit = t < depth
    img[hit] = floor_col[hit]
    depth[hit] = t[hit]
    # 천장 y=3
    t = np.where(dy > 1e-6, (3.0 - cam_y) / np.where(dy > 1e-6, dy, 1), np.inf)
    hit = t < depth
    px, pz = eye_x + dx * t, dz * t
    stripes = ((np.floor(pz * 2)) % 2).astype(bool)
    col = np.where(stripes[..., None], (0.55, 0.58, 0.65), (0.45, 0.48, 0.55))
    img[hit] = col[hit]
    depth[hit] = t[hit]
    # 앞 벽 z=8
    t = np.where(dz > 1e-6, 8.0 / dz, np.inf)
    hit = t < depth
    px, py = eye_x + dx * t, cam_y + dy * t
    bricks = ((np.floor(px * 2) + np.floor(py * 4)) % 2).astype(bool)
    col = np.where(bricks[..., None], (0.62, 0.40, 0.35), (0.52, 0.32, 0.28))
    img[hit] = col[hit]
    depth[hit] = t[hit]
    # 양 옆 벽 x=±5
    for sx in (-5.0, 5.0):
        denom = np.where(np.abs(dx) > 1e-6, dx, 1)
        t = np.where((sx - eye_x) / denom > 0, (sx - eye_x) / denom, np.inf)
        hit = t < depth
        py, pz = cam_y + dy * t, dz * t
        panels = ((np.floor(pz) + np.floor(py * 2)) % 2).astype(bool)
        col = np.where(panels[..., None], (0.40, 0.55, 0.45), (0.30, 0.45, 0.35))
        img[hit] = col[hit]
        depth[hit] = t[hit]
    # 떠 있는 구 몇 개
    rng = np.random.default_rng(seed)
    for _ in range(6):
        c = np.array([rng.uniform(-2.5, 2.5), rng.uniform(0.6, 2.4), rng.uniform(2.0, 6.0)])
        r = rng.uniform(0.25, 0.55)
        colr = rng.uniform(0.3, 1.0, 3)
        o = np.array([eye_x, cam_y, 0.0])
        oc = o - c
        d = np.stack([dx, dy, dz], -1)
        b = (d * oc).sum(-1)
        cc = oc @ oc - r * r
        disc = b * b - cc
        t = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
        t = np.where(t > 0, t, np.inf)
        hit = t < depth
        p = o + d * t[..., None]
        nrm = (p - c) / r
        shade = np.clip(0.4 + 0.6 * (nrm @ np.array([0.3, 0.8, -0.5])), 0.2, 1.0)
        img[hit] = colr[None, :] * shade[hit][:, None]
        depth[hit] = t[hit]
    # 간단한 거리 안개
    fog = np.clip(1 - depth / 30.0, 0.3, 1.0)
    img = img * fog[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def make_sbs(
    width_per_eye: int = 1920,
    height: int = 1080,
    hfov_deg: float = 80.0,
    baseline_m: float = 0.065,
    seed: int = 0,
) -> np.ndarray:
    left = _render(width_per_eye, height, hfov_deg, -baseline_m / 2, seed)
    right = _render(width_per_eye, height, hfov_deg, +baseline_m / 2, seed)
    return np.concatenate([left, right], axis=1)
