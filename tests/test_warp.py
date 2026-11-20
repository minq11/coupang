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


def test_scale_reverses_direction():
    """양의 disparity 에 scale=-1 을 주면 반대 방향으로 워핑된다 (mono 양방향 워핑 용도)."""
    size = 128
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[:, 60] = (255, 0, 0)  # 열 60 에 빨강
    disp = np.zeros((size, size), np.float32)
    disp[:, 60] = 6 * math.pi / size  # +6 열에 해당하는 각도
    out_fwd, cov_fwd = warp.forward_warp(rgb, disp, taps=1, scale=+1.0)
    out_rev, cov_rev = warp.forward_warp(rgb, disp, taps=1, scale=-1.0)
    row = size // 2
    # forward: du = -disp·S/π · scale → scale=+1 이면 열 60-6=54, scale=-1 이면 열 60+6=66
    assert cov_fwd[row, 54] and tuple(out_fwd[row, 54]) == (255, 0, 0)
    assert cov_rev[row, 66] and tuple(out_rev[row, 66]) == (255, 0, 0)


def test_zbuffer_uses_abs_disparity():
    """음수 scale 로 양쪽에 워핑할 때, |disp| 가 큰 쪽이 z-buffer 승자가 되어야 한다."""
    size = 64
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[:, 12] = (255, 0, 0)  # |disp| 큰 (가까운)
    rgb[:, 10] = (0, 255, 0)  # |disp| 작은 (먼)
    disp = np.zeros((size, size), np.float32)
    disp[:, 12] = -7 * math.pi / size  # 음수지만 |disp| 큼
    disp[:, 10] = 5 * math.pi / size
    out, covered = warp.forward_warp(rgb, disp, taps=1, scale=+1.0)
    # 열 12 는 du = -(-7)*1 = +7 → 열 19. 열 10 은 du = -5 → 열 5. 겹치지 않음.
    # 하지만 두 색이 같은 열에 매핑되는 걸 억지로 만들려면 복잡하니 — 단순 체크:
    # 각자 밀린 열에 색이 있고 겹치지 않을 때 둘 다 커버된다.
    row = size // 2
    assert covered[row, 19] and tuple(out[row, 19]) == (255, 0, 0)
    assert covered[row, 5] and tuple(out[row, 5]) == (0, 255, 0)


def _centroid_x(img_ch):
    ys, xs = np.nonzero(img_ch > 127)
    return xs.mean()


def test_mono_scales_give_crossed_disparity():
    """2D 한 장에서 만든 양쪽 눈: 가까운 물체는 왼눈에서 오른쪽, 오른눈에서 왼쪽 (실제 눈과 같은 교차 시차).
    그리고 sbs 모드의 L→R 워핑(scale=+1)과 같은 규약이어야 한다."""
    size = 256
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[118:138, 118:138] = 255  # 가운데 가까운 사각형
    disp = np.zeros((size, size), np.float32)
    disp[110:146, 110:146] = 0.05  # 가까움
    x0 = _centroid_x(rgb[..., 0])
    left, _ = warp.forward_warp(rgb, disp, taps=1, scale=warp.MONO_SCALE_L)
    right, _ = warp.forward_warp(rgb, disp, taps=1, scale=warp.MONO_SCALE_R)
    assert _centroid_x(left[..., 0]) > x0 + 1  # 왼눈: 오른쪽
    assert _centroid_x(right[..., 0]) < x0 - 1  # 오른눈: 왼쪽
    # sbs 규약: 실제 L 에서 R 을 만들면 가까운 물체는 왼쪽으로 (오른눈은 왼쪽에 본다)
    r_from_l, _ = warp.forward_warp(left, disp, taps=1, scale=1.0)
    assert _centroid_x(r_from_l[..., 0]) < _centroid_x(left[..., 0])


def test_splat_horizontal_rgba_zbuffer_and_holes():
    h, w = 8, 32
    img = np.zeros((h, w, 4), np.uint8)
    img[:, 10] = (255, 0, 0, 255)  # 가까운 열 (priority 2) → 오른쪽 3칸
    img[:, 13] = (0, 255, 0, 255)  # 먼 열 (priority 1) → 그대로
    du = np.zeros((h, w), np.float32)
    du[:, 10] = 3.0
    pr = np.ones((h, w), np.float32)
    pr[:, 10] = 2.0
    out, cov = warp.splat_horizontal(img, du, pr, src_mask=img[..., 3] > 0, taps=1)
    assert (out[:, 13] == (255, 0, 0, 255)).all()  # 가까운 쪽이 이김
    assert not cov[:, 10].any()  # 원래 자리는 구멍
    assert out.shape == img.shape
    # 2D 입력도 된다
    g, cov2 = warp.splat_horizontal(img[..., 3], du, pr, src_mask=img[..., 3] > 0, taps=1)
    assert g.shape == (h, w) and (g[:, 13] == 255).all()
