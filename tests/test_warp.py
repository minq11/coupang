import math

import numpy as np

from vr180.geometry import sphere, warp


def test_single_depth_plane_shift():
    size = 256
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    disp = np.full((size, size), 0.02, np.float32)  # 일정한 각도 disparity
    out, covered = warp.forward_warp(rgb, disp, taps=1)
    _, phi = sphere.equi_angle_grid(size)
    for row in (size // 2, size // 4, 3 * size // 4):
        du = -0.02 / math.cos(phi[row, 0]) * size / math.pi
        shift = int(round(du))
        src = rgb[row]
        dst = out[row]
        # 목적지 열 x = 원본 열 x - |shift|
        n = size - abs(shift)
        if shift < 0:
            assert np.array_equal(dst[:n], src[-shift : -shift + n])
            assert covered[row, :n].all()
            assert not covered[row, n:].any()  # 오른쪽 끝은 hole


def test_zbuffer_near_wins():
    size = 64
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[:, 12] = (255, 0, 0)  # 가까운 점 (disparity 큼): 열 12 → 열 5
    rgb[:, 10] = (0, 255, 0)  # 먼 점: 열 10 → 열 5
    disp = np.zeros((size, size), np.float32)
    disp[:, 12] = 7 * math.pi / size
    disp[:, 10] = 5 * math.pi / size
    out, covered = warp.forward_warp(rgb, disp, taps=1)
    # 중앙 행(φ≈0) 에서 확인. 극 쪽 행은 1/cosφ 때문에 다른 열로 간다.
    for row in (size // 2 - 1, size // 2):
        assert covered[row, 5]
        assert tuple(out[row, 5]) == (255, 0, 0)


def test_hole_mask_region():
    cov = np.ones((8, 8), bool)
    cov[2, 3] = False
    region = np.zeros((8, 8), bool)
    region[2, :] = True
    m = warp.hole_mask(cov, region)
    assert m[2, 3] == 255 and m.sum() == 255
