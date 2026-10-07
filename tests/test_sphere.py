import math

import numpy as np

from vr180.geometry import sphere


def test_equi_pixel_roundtrip():
    size = 1024
    rng = np.random.default_rng(0)
    u = rng.uniform(0, size - 1, 10000)
    v = rng.uniform(0, size - 1, 10000)
    th, ph = sphere.equi_pixel_to_angles(u, v, size)
    u2, v2 = sphere.angles_to_equi_pixel(th, ph, size)
    assert np.max(np.abs(u - u2)) < 1e-6
    assert np.max(np.abs(v - v2)) < 1e-6


def test_angles_dir_roundtrip():
    rng = np.random.default_rng(1)
    th = rng.uniform(-math.pi / 2 + 1e-3, math.pi / 2 - 1e-3, 10000)
    ph = rng.uniform(-math.pi / 2 + 1e-3, math.pi / 2 - 1e-3, 10000)
    d = sphere.angles_to_dir(th, ph)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-9)
    th2, ph2 = sphere.dir_to_angles(d)
    assert np.max(np.abs(th - th2)) < 1e-6
    assert np.max(np.abs(ph - ph2)) < 1e-6


def test_equi_grid_corners():
    size = 4
    th, ph = sphere.equi_angle_grid(size)
    # 왼쪽 열은 -90° 쪽, 위쪽 행은 +90° 쪽
    assert th[0, 0] < 0 and th[0, -1] > 0
    assert ph[0, 0] > 0 and ph[-1, 0] < 0
    assert abs(th[0, 0] + th[0, -1]) < 1e-12  # 대칭


def test_pinhole_roundtrip():
    w, h, hfov = 1920, 1080, math.radians(80)
    rng = np.random.default_rng(2)
    px = rng.uniform(0, w - 1, 5000)
    py = rng.uniform(0, h - 1, 5000)
    d = sphere.pinhole_to_dir(px, py, w, h, hfov)
    px2, py2, inside = sphere.dir_to_pinhole(d, w, h, hfov)
    assert inside.all()
    assert np.max(np.abs(px - px2)) < 1e-6
    assert np.max(np.abs(py - py2)) < 1e-6


def test_pinhole_fov_edges():
    w, h, hfov = 1920, 1080, math.radians(80)
    # 이미지 중앙 = forward, 왼쪽 가장자리 = -hfov/2
    d = sphere.pinhole_to_dir(np.array([w / 2 - 0.5]), np.array([h / 2 - 0.5]), w, h, hfov)
    th, ph = sphere.dir_to_angles(d)
    assert abs(th[0]) < 1e-9 and abs(ph[0]) < 1e-9
    d = sphere.pinhole_to_dir(np.array([-0.5]), np.array([h / 2 - 0.5]), w, h, hfov)
    th, _ = sphere.dir_to_angles(d)
    assert abs(th[0] + hfov / 2) < 1e-9
    vfov = sphere.vfov_from_hfov(hfov, w, h)
    d = sphere.pinhole_to_dir(np.array([w / 2 - 0.5]), np.array([-0.5]), w, h, hfov)
    _, ph = sphere.dir_to_angles(d)
    assert abs(ph[0] - vfov / 2) < 1e-9


def test_rotation_forward():
    for yaw, pitch in [(0.3, 0.0), (0.0, -0.7), (-1.0, 0.4)]:
        rot = sphere.rotation_yaw_pitch(yaw, pitch)
        fwd = sphere.cam_to_world(np.array([0.0, 0.0, 1.0]), rot)
        assert np.allclose(fwd, sphere.angles_to_dir(yaw, pitch), atol=1e-12)
        back = sphere.world_to_cam(fwd, rot)
        assert np.allclose(back, [0, 0, 1], atol=1e-12)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)


def test_angular_disparity():
    f = 1000.0
    d = sphere.pixel_disparity_to_angular(np.array([10.0]), np.array([0.0]), f)
    assert abs(d[0] - 0.01) < 1e-9
    d2 = sphere.pixel_disparity_to_angular(np.array([10.0]), np.array([1.0]), f)
    assert abs(d2[0] - 0.005) < 1e-9
    d3 = sphere.pixel_disparity_to_angular(np.array([10.0]), np.array([1.0]), f, exact=False)
    assert abs(d3[0] - 0.01) < 1e-9
