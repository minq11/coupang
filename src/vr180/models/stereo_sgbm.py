"""OpenCV StereoSGBM. 모델/GPU 없이 Stage 4 를 돌리기 위한 기본 구현."""

from __future__ import annotations

import cv2
import numpy as np

from vr180.models.base import ModelInfo, StereoMatcher


class SGBMMatcher(StereoMatcher):
    def _load(self) -> None:
        pass

    def _unload(self) -> None:
        pass

    def info(self) -> ModelInfo:
        return ModelInfo(name="opencv_sgbm", device="cpu")

    def disparity(self, left: np.ndarray, right: np.ndarray, max_disparity: int = 256):
        num = int(np.ceil(max_disparity / 16.0) * 16)
        num = max(16, min(num, 1024))
        block = 5
        gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num,
            blockSize=block,
            P1=8 * 3 * block * block,
            P2=32 * 3 * block * block,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        d = sgbm.compute(gl, gr).astype(np.float32) / 16.0
        valid = d > 0
        # 왼쪽 가장자리 num 픽셀은 SGBM 이 채우지 못한다 → 가까운 유효값으로 메움
        d = _fill_invalid(d, valid)
        return d.astype(np.float32), valid


def _fill_invalid(d: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from vr180.util.fill import nearest_fill

    return nearest_fill(d, valid)
