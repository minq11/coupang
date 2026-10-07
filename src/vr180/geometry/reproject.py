"""remap 테이블 생성: 핀홀 ↔ half-equirect, 원근 타일 ↔ half-equirect.

모든 테이블은 cv2.remap(src, map_x, map_y) 에 바로 넣을 수 있는 float32 쌍이다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from vr180.geometry import sphere

INTERP = {
    "lanczos": cv2.INTER_LANCZOS4,
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def pinhole_to_equi_maps(
    size: int, width: int, height: int, hfov_rad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """equirect 캔버스의 각 픽셀이 핀홀 이미지의 어느 좌표를 샘플링하는지.

    반환: map_x, map_y (S×S float32), valid (S×S bool).
    """
    theta, phi = sphere.equi_angle_grid(size)
    d = sphere.angles_to_dir(theta, phi)
    px, py, inside = sphere.dir_to_pinhole(d, width, height, hfov_rad)
    map_x = np.where(inside, px, -10.0).astype(np.float32)
    map_y = np.where(inside, py, -10.0).astype(np.float32)
    return map_x, map_y, inside


def remap_to_equi(
    img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray, interpolation: str = "lanczos"
) -> np.ndarray:
    """핀홀 RGB(A) 이미지를 equirect RGBA 로. 원본 밖은 알파 0."""
    if img.ndim == 2:
        img = img[..., None]
    rgb = img[..., :3]
    out = cv2.remap(rgb, map_x, map_y, INTERP[interpolation], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if out.ndim == 2:
        out = out[..., None]
    alpha = (valid.astype(np.uint8) * 255)[..., None]
    out = np.where(valid[..., None], out, 0)
    return np.concatenate([out.astype(np.uint8), alpha], axis=-1)


def remap_float_to_equi(
    arr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray, fill: float = 0.0
) -> np.ndarray:
    """float32 스칼라 맵(disparity 등)을 equirect 로. 선형 보간. 원본 밖은 fill."""
    arr = np.asarray(arr, dtype=np.float32)
    out = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.where(valid, out, np.float32(fill)).astype(np.float32)


# ----------------------------------------------------------------------------- 원근 타일
class TileCamera:
    """equirect 위에 놓인 가상 핀홀 카메라 (정사각 타일)."""

    def __init__(self, yaw_deg: float, pitch_deg: float, fov_deg: float, res: int):
        self.yaw_deg = float(yaw_deg)
        self.pitch_deg = float(pitch_deg)
        self.fov_deg = float(fov_deg)
        self.res = int(res)
        self.rot = sphere.rotation_yaw_pitch(math.radians(yaw_deg), math.radians(pitch_deg))
        self.name = f"y{int(round(yaw_deg)):+d}_p{int(round(pitch_deg)):+d}"

    # equirect → 타일 (타일을 렌더링할 때: 타일 픽셀마다 equirect 좌표)
    def equi_sample_maps(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.mgrid[0 : self.res, 0 : self.res]
        d_cam = sphere.pinhole_to_dir(xs, ys, self.res, self.res, math.radians(self.fov_deg))
        d_world = sphere.cam_to_world(d_cam.reshape(-1, 3), self.rot).reshape(self.res, self.res, 3)
        theta, phi = sphere.dir_to_angles(d_world)
        u, v = sphere.angles_to_equi_pixel(theta, phi, size)
        # 반구 뒤쪽(θ 가 ±90° 를 넘는 경우)은 캔버스 밖 → remap 이 border 처리
        behind = d_world[..., 2] <= 0
        u = np.where(behind, -10.0, u)
        v = np.where(behind, -10.0, v)
        return u.astype(np.float32), v.astype(np.float32)

    def render_from_equi(
        self, pano_rgba: np.ndarray, interpolation: str = "lanczos"
    ) -> tuple[np.ndarray, np.ndarray]:
        """equirect RGBA → 타일 RGB (uint8) 와 알파 (float32 0..1)."""
        size = pano_rgba.shape[0]
        mx, my = self.equi_sample_maps(size)
        rgb = cv2.remap(
            pano_rgba[..., :3], mx, my, INTERP[interpolation], borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        alpha = cv2.remap(
            pano_rgba[..., 3], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return rgb, alpha.astype(np.float32) / 255.0

    def render_float_from_equi(self, arr: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        size = arr.shape[0]
        mx, my = self.equi_sample_maps(size)
        return cv2.remap(
            np.asarray(arr, dtype=np.float32), mx, my, interpolation, borderMode=cv2.BORDER_REPLICATE
        )

    # 타일 → equirect (타일 결과를 써넣을 때: equirect 픽셀마다 타일 좌표)
    def tile_sample_maps(self, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta, phi = sphere.equi_angle_grid(size)
        d_world = sphere.angles_to_dir(theta, phi).reshape(-1, 3)
        d_cam = sphere.world_to_cam(d_world, self.rot).reshape(size, size, 3)
        px, py, inside = sphere.dir_to_pinhole(d_cam, self.res, self.res, math.radians(self.fov_deg))
        mx = np.where(inside, px, -10.0).astype(np.float32)
        my = np.where(inside, py, -10.0).astype(np.float32)
        return mx, my, inside

    def feather_weight(self, size: int, feather_px: int) -> tuple[np.ndarray, np.ndarray]:
        """타일이 equirect 에서 덮는 영역(inside) 과 가장자리 페더 가중치 (0..1, 타일 픽셀 단위 페더)."""
        mx, my, inside = self.tile_sample_maps(size)
        if feather_px <= 0:
            return inside, inside.astype(np.float32)
        # 타일 안에서 가장자리까지의 거리 (타일 픽셀 단위)
        dist = np.minimum(np.minimum(mx, my), np.minimum(self.res - 1 - mx, self.res - 1 - my))
        w = np.clip(dist / float(feather_px), 0.0, 1.0)
        w = np.where(inside, w, 0.0).astype(np.float32)
        return inside, w

    def splat_to_equi(
        self, tile_rgb: np.ndarray, size: int, interpolation: str = "lanczos"
    ) -> tuple[np.ndarray, np.ndarray]:
        """타일 RGB 를 equirect 캔버스로 (역방향 remap). 반환: rgb(S×S×3), inside."""
        mx, my, inside = self.tile_sample_maps(size)
        out = cv2.remap(
            tile_rgb, mx, my, INTERP[interpolation], borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return out, inside

    def splat_float_to_equi(self, tile_arr: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        mx, my, inside = self.tile_sample_maps(size)
        out = cv2.remap(
            np.asarray(tile_arr, dtype=np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return out, inside


def make_tile_grid(
    yaws_deg, pitches_deg, fov_deg: float, res: int, skip_center: bool = True
) -> list[TileCamera]:
    """설계서 순서: 중앙 → 좌/우 → 상/하 → 모서리."""
    tiles: list[TileCamera] = []
    center = TileCamera(0, 0, fov_deg, res)
    if not skip_center:
        tiles.append(center)
    sides = [TileCamera(y, 0, fov_deg, res) for y in yaws_deg if y != 0]
    updown = [TileCamera(0, p, fov_deg, res) for p in pitches_deg if p != 0]
    corners = [TileCamera(y, p, fov_deg, res) for p in pitches_deg if p != 0 for y in yaws_deg if y != 0]
    tiles += sides + updown + corners
    return tiles
