"""유효하지 않은 픽셀을 가장 가까운 유효 픽셀 값으로 채운다 (float 맵·RGB 공용)."""

from __future__ import annotations

import cv2
import numpy as np


def nearest_labels(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """valid(bool H×W) 에 대해 각 픽셀의 '가장 가까운 유효 픽셀' 라벨 맵과,
    라벨 → 평탄 인덱스 테이블을 돌려준다. 한 번 계산해 여러 채널에 재사용."""
    src = (~valid).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    flat_valid_idx = np.flatnonzero(valid.ravel())
    lut_idx = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    lut_idx[labels[valid]] = flat_valid_idx
    return labels, lut_idx


def nearest_fill(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """arr: H×W float, valid: H×W bool. 유효하지 않은 곳에 최근접 유효값 복사."""
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return arr.copy()
    if not valid.any():
        return np.zeros_like(arr)
    labels, lut_idx = nearest_labels(valid)
    out = arr.ravel()[lut_idx[labels]]
    out[valid] = arr[valid]
    return out


def nearest_fill_rgb(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """img: H×W×C (uint8/float), valid: H×W bool. 채널마다 최근접 유효 픽셀 값으로 채운다."""
    img = np.asarray(img)
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return img.copy()
    if not valid.any():
        return np.zeros_like(img)
    labels, lut_idx = nearest_labels(valid)
    flat = img.reshape(-1, img.shape[-1])
    out = flat[lut_idx[labels]]
    out[valid] = img[valid]
    return out.astype(img.dtype)
