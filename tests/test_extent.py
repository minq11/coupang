"""생성 범위(extent) 모듈 단위 테스트."""

from __future__ import annotations

import numpy as np

from vr180.geometry import extent


def test_compute_hv_margin_rule():
    # 80° 16:9 → vfov ≈ 2·atan(tan(40°)·9/16) ≈ ... Stage 1 이 계산. 여기서는 명시해서 테스트.
    h, v = extent.compute_extent_hv(80.0, 50.0, {"extent_margin_h_deg": 20, "extent_margin_v_deg": 25})
    assert abs(h - 120.0) < 1e-6
    assert abs(v - 100.0) < 1e-6


def test_compute_hv_fixed_overrides_margin():
    h, v = extent.compute_extent_hv(80.0, 50.0, {"extent_h_deg": 150, "extent_v_deg": 140})
    assert h == 150 and v == 140


def test_compute_hv_clamps_to_180():
    h, v = extent.compute_extent_hv(170.0, 170.0, {"extent_margin_h_deg": 50, "extent_margin_v_deg": 50})
    assert h <= 180 and v <= 180


def test_weight_center_is_1_edge_is_0():
    S = 256
    mask, w = extent.extent_weight(S, 120, 100, fade_deg=10)
    # 중앙 (θ=0, φ=0) 은 S/2 근처. S×S 그리드에서 (u+0.5)/S·π - π/2 가 u=S/2 에서 ≈ 0.
    cx = S // 2
    assert w[cx, cx] > 0.99
    # 상단 가장자리 (θ=0, φ=90°, 밖) 는 0. v=100 이므로 ±50° 가 축. φ=90° 는 r=1.8.
    assert w[0, cx] < 1e-3


def test_weight_fully_inside_at_margin_minus_fade():
    S = 512
    h, v, f = 120.0, 100.0, 10.0
    mask, w = extent.extent_weight(S, h, v, fade_deg=f)
    # r ≤ 1−f_ratio 는 w = 1. 중앙 세로 열에서 φ 가 ±(v/2)·(1−f/50) 안이면 w == 1.
    # 간단히 (θ=0, φ=0..30°) 를 샘플링하면 반드시 1.
    center_col = S // 2
    for v_px in (S // 2, S // 2 - 30, S // 2 - 60):
        assert w[v_px, center_col] > 0.99


def test_fully_disabled_180_180():
    S = 128
    mask, w = extent.extent_weight(S, 180.0, 180.0, fade_deg=0.0)
    assert mask.all()
    assert np.all(w == 1.0)


def test_outside_corners_at_120_100():
    """120×100 에서 모서리 (θ, φ)=(±90°, ±90°) 는 범위 밖이어야 한다."""
    S = 256
    mask, w = extent.extent_weight(S, 120.0, 100.0, fade_deg=0.0)
    assert not mask[0, 0]
    assert not mask[0, S - 1]
    assert not mask[S - 1, 0]
    assert not mask[S - 1, S - 1]
