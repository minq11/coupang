"""유효하지 않은 픽셀을 가장 가까운 유효 픽셀 값으로 채운다 (float 맵용)."""

from __future__ import annotations

import cv2
import numpy as np


def nearest_fill(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """arr: H×W float, valid: H×W bool. 유효하지 않은 곳에 최근접 유효값 복사."""
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return arr.copy()
    if not valid.any():
        return np.zeros_like(arr)
    # distanceTransform 은 0 픽셀까지의 거리를 재므로, 유효 픽셀을 0 으로 둔다
    src = (~valid).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # 라벨 k 는 k 번째(래스터 순) 0-픽셀. 유효 픽셀의 라벨 → 값 테이블
    flat_valid_idx = np.flatnonzero(valid.ravel())
    lut = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    lab_valid = labels[valid]
    lut[lab_valid] = arr.ravel()[flat_valid_idx]
    out = lut[labels]
    out[valid] = arr[valid]
    return out
