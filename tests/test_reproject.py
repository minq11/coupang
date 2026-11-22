import math

import numpy as np

from vr180.geometry import reproject, sphere


def _line_image(w, h, col=None, row=None, thick=3):
    img = np.zeros((h, w, 3), np.uint8)
    if col is not None:
        img[:, col - thick // 2 : col + thick // 2 + 1] = (255, 0, 0)
    if row is not None:
        img[row - thick // 2 : row + thick // 2 + 1, :] = (0, 255, 0)
    return img


def test_valid_mask_extent_matches_fov():
    size, w, h = 512, 640, 360
    hfov = math.radians(80)
    mx, my, valid = reproject.pinhole_to_equi_maps(size, w, h, hfov)
    th, ph = sphere.equi_angle_grid(size)
    mid = size // 2
    row = valid[mid]
    cols = np.flatnonzero(row)
    assert abs(math.degrees(th[mid, cols[0]]) + 40) < 0.5
    assert abs(math.degrees(th[mid, cols[-1]]) - 40) < 0.5
    vfov = sphere.vfov_from_hfov(hfov, w, h)
    rows = np.flatnonzero(valid[:, mid])
    assert abs(ph[rows[0], mid] - vfov / 2) < math.radians(0.5)
    assert abs(ph[rows[-1], mid] + vfov / 2) < math.radians(0.5)


def test_vertical_line_stays_vertical_and_horizontal_line_curves():
    size, w, h = 1024, 640, 360
    hfov = math.radians(90)
    f = sphere.focal_px(w, hfov)
    col, row = 480, 60  # 오른쪽 열, 위쪽 행
    img = _line_image(w, h, col=col, row=row)
    mx, my, valid = reproject.pinhole_to_equi_maps(size, w, h, hfov)
    out = reproject.remap_to_equi(img, mx, my, valid, "nearest")
    th, ph = sphere.equi_angle_grid(size)
    # 빨간(수직선) 픽셀: θ 가 일정해야 한다. x_norm = tanθ
    red = (out[..., 0] > 200) & (out[..., 1] < 50)
    x_norm = (col + 0.5 - w / 2) / f
    th_expect = math.atan(x_norm)
    assert red.sum() > 50
    assert np.max(np.abs(th[red] - th_expect)) < math.radians(0.6)
    # 초록(수평선) 픽셀: φ = atan(y_norm · cosθ) 곡선 위에 있어야 한다
    green = (out[..., 1] > 200) & (out[..., 0] < 50)
    y_norm = (h / 2 - (row + 0.5)) / f
    ph_expect = np.arctan(y_norm * np.cos(th[green]))
    assert green.sum() > 50
    assert np.max(np.abs(ph[green] - ph_expect)) < math.radians(0.6)
    # 곡선인지 (φ 가 열마다 다름)
    assert np.ptp(ph[green]) > math.radians(2)


def test_checkerboard_roundtrip_through_tile():
    """equirect 에 올린 체커보드를 타일로 꺼냈다가 다시 넣으면 원래 자리에 온다."""
    size = 256
    yy, xx = np.mgrid[0:size, 0:size]
    checker = (((yy // 16) + (xx // 16)) % 2 * 255).astype(np.uint8)
    pano = np.dstack([checker, checker, checker, np.full((size, size), 255, np.uint8)])
    tile = reproject.TileCamera(30, -20, 90, 256)
    rgb, alpha = tile.render_from_equi(pano, "nearest")
    assert alpha.min() > 0.99
    back, inside = tile.splat_to_equi(rgb, size, "nearest")
    # 타일 안쪽 (가장자리 제외) 에서 일치 비율
    _, w = tile.feather_weight(size, 12)
    core = inside & (w >= 1.0)
    diff = np.abs(back[core][:, 0].astype(int) - pano[core][:, 0].astype(int))
    assert (diff < 30).mean() > 0.9


def test_tile_grid_covers_hemisphere():
    size = 256
    tiles = reproject.make_tile_grid([-60, 0, 60], [-45, 0, 45], 90, 64, skip_center=False)
    assert len(tiles) == 9
    cov = np.zeros((size, size), bool)
    for t in tiles:
        _, _, inside = t.tile_sample_maps(size)
        cov |= inside
    th, ph = sphere.equi_angle_grid(size)
    # 극 바로 근처(|φ| > 85°) 와 가장자리는 제외하고 거의 전부 덮여야 한다
    region = (np.abs(ph) < math.radians(85)) & (np.abs(th) < math.radians(88))
    assert cov[region].mean() > 0.995


def test_tilt_is_true_rotation_roundtrip():
    """pitch/roll 은 진짜 회전: equirect 픽셀 → 핀홀 좌표 → 방향 → 역회전 하면 원래 (θ, φ) 로 돌아온다.
    (예전 φ+pitch 방식은 가장자리에서 이 검사에 걸린다.)"""
    size, w, h = 256, 640, 480
    hfov = math.radians(80)
    for pitch, roll in [(math.radians(25), 0.0), (0.0, math.radians(8)), (math.radians(-15), math.radians(-5))]:
        mx, my, valid = reproject.pinhole_to_equi_maps(size, w, h, hfov, pitch, roll)
        th, ph = sphere.equi_angle_grid(size)
        d_cam = sphere.pinhole_to_dir(mx[valid], my[valid], w, h, hfov)
        d_world = d_cam @ sphere.photo_tilt_matrix(pitch, roll)  # M^T 의 역 = M (직교)
        t2, p2 = sphere.dir_to_angles(d_world)
        assert np.abs(t2 - th[valid]).max() < 1e-6
        assert np.abs(p2 - ph[valid]).max() < 1e-6


def test_pitch_puts_equator_above_center_and_roll_tilts_horizon():
    size, w, h = 512, 640, 480
    hfov = math.radians(80)
    f = sphere.focal_px(w, hfov)
    mx, my, _ = reproject.pinhole_to_equi_maps(size, w, h, hfov, math.radians(20))
    c = size // 2
    # VR 정면 (θ≈0, φ≈0) 이 사진 중앙보다 f·tan20° 위를 샘플링
    assert abs((my[c, c] + 0.5) - (h / 2 - f * math.tan(math.radians(20)))) < 2.0
    # roll +8°: 사진 속 수평선이 반시계로 기운 것 → VR 수평선이 사진 오른쪽에서 더 위(py 작음)
    mx, my, _ = reproject.pinhole_to_equi_maps(size, w, h, hfov, 0.0, math.radians(8))
    th, _ = sphere.equi_angle_grid(size)
    a, b = np.searchsorted(th[c], math.radians(-20)), np.searchsorted(th[c], math.radians(20))
    slope = -(my[c, b] - my[c, a]) / (mx[c, b] - mx[c, a])
    assert abs(math.degrees(math.atan(slope)) - 8) < 0.3


def test_pitch_guide_matches_legacy_row_without_roll():
    from vr180.gui.common import draw_pitch_guide

    rgb = np.zeros((480, 640, 3), np.uint8)
    out = draw_pitch_guide(rgb, 80, 15)
    f = 640 / (2 * math.tan(math.radians(40)))
    row = round(240 - f * math.tan(math.radians(15)))
    ys = np.flatnonzero((out[:, 320] == (255, 220, 0)).all(axis=1))
    assert ys.size and abs(np.median(ys) - row) <= 2
    out = draw_pitch_guide(rgb, 80, 0, 6)
    yl = np.flatnonzero(out[:, 100, 0] > 200).mean()
    yr = np.flatnonzero(out[:, 540, 0] > 200).mean()
    assert yr < yl  # roll + : 오른쪽이 더 위
