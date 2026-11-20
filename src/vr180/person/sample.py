"""인물 AR 합성 샘플 — 모델 없이 파이프라인을 끝까지 돌려 보기 위한 장면.

배경(하늘·바닥, 채도 낮음) 위에 채도 높은 '사람' 두 명 (머리 원 + 몸 캡슐).
영상은 A 가 제자리에서 조금씩 움직이고, B 가 키프레임 사이에 오른쪽에서 걸어 들어온다.
make_*() 은 (rgb, 사람별 정답 마스크 목록, 정답 inverse depth) 를 돌려준다.
"""

from __future__ import annotations

import cv2
import numpy as np

PERSON_COLORS = [(220, 40, 160), (30, 170, 230)]  # 채도 높은 색 (테스트용 가짜 검출기가 색으로 찾는다)


def _background(w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.float32)
    horizon = int(h * 0.55)
    t = np.linspace(0, 1, horizon)[:, None]
    img[:horizon] = (np.array([150, 160, 175]) * (1 - t) + np.array([190, 190, 185]) * t)[:, None, :]
    rows = h - horizon
    yy, xx = np.mgrid[0:rows, 0:w]
    tile = (((xx // max(4, w // 24)) + (yy // max(4, rows // 6))) % 2).astype(np.float32)
    img[horizon:] = (110 + 25 * tile)[..., None] * np.array([1.0, 0.97, 0.92])
    return img


def _person(h: int, w: int, cx: float, feet_y: float, height: float) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    head_r = height * 0.09
    body_top = feet_y - height + 2 * head_r
    cv2.circle(m, (int(cx), int(feet_y - height + head_r)), int(head_r), 255, -1)
    bw = height * 0.13
    neck = height * 0.035  # 머리와 몸을 잇는 목 (실제 사람 마스크처럼 한 덩어리가 되게)
    cv2.rectangle(
        m, (int(cx - neck), int(feet_y - height + head_r)), (int(cx + neck), int(body_top + head_r)), 255, -1
    )
    cv2.rectangle(
        m, (int(cx - bw), int(body_top + head_r * 0.6)), (int(cx + bw), int(feet_y - height * 0.35)), 255, -1
    )
    leg = height * 0.05
    cv2.rectangle(
        m,
        (int(cx - bw * 0.8), int(feet_y - height * 0.36)),
        (int(cx - bw * 0.8 + 2 * leg), int(feet_y)),
        255,
        -1,
    )
    cv2.rectangle(
        m,
        (int(cx + bw * 0.8 - 2 * leg), int(feet_y - height * 0.36)),
        (int(cx + bw * 0.8), int(feet_y)),
        255,
        -1,
    )
    arm = height * 0.04
    cv2.rectangle(
        m,
        (int(cx - bw - 2 * arm), int(body_top + head_r)),
        (int(cx - bw), int(feet_y - height * 0.45)),
        255,
        -1,
    )
    cv2.rectangle(
        m,
        (int(cx + bw), int(body_top + head_r)),
        (int(cx + bw + 2 * arm), int(feet_y - height * 0.45)),
        255,
        -1,
    )
    return m > 0


def _render(
    w: int, h: int, people: list[tuple[float, float, float]]
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """people: (cx, feet_y, height) 목록. 뒤(작은 키)부터 그린다."""
    img = _background(w, h)
    inv = np.full((h, w), 0.1, np.float32)  # 배경 = 멀다
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    inv[int(h * 0.55) :] = 0.1 + 0.3 * (yy[int(h * 0.55) :] - h * 0.55) / (
        h * 0.45
    )  # 바닥은 아래로 갈수록 가까움
    masks = []
    order = sorted(range(len(people)), key=lambda i: people[i][2])
    for i in order:
        cx, fy, ht = people[i]
        m = _person(h, w, cx, fy, ht)
        color = np.array(PERSON_COLORS[i % len(PERSON_COLORS)], np.float32)
        shade = 0.75 + 0.25 * np.clip((yy - (fy - ht)) / max(ht, 1), 0, 1)  # 아래로 갈수록 밝게 (채도 유지)
        img[m] = color[None, :] * shade[m][:, None]
        near = 0.5 + 0.5 * ht / h  # 키가 클수록 가깝다
        inv[m] = (
            near + 0.05 * (np.abs(np.mgrid[0:h, 0:w][1] - cx) < ht * 0.08)[m]
        )  # 몸 가운데가 조금 더 가까움
        for prev in masks:  # 앞 사람에 가려진 부분은 뒷사람 마스크에서 뺀다
            prev &= ~m
        masks.append(m)
    # masks 는 그린 순서. 원래 순서로 되돌린다.
    out = [None] * len(people)
    for k, i in enumerate(order):
        out[i] = masks[k]
    return np.clip(img, 0, 255).astype(np.uint8), out, inv


def make_person_image(w: int = 640, h: int = 360) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    return _render(w, h, [(w * 0.38, h * 0.92, h * 0.75), (w * 0.68, h * 0.80, h * 0.5)])


def make_person_video(
    w: int = 320, h: int = 180, n: int = 60, enter_frame: int = 20
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """A 는 처음부터 있고 조금씩 흔들린다. B 는 enter_frame 에 오른쪽 밖에서 들어와 왼쪽으로 걷는다."""
    frames, gts = [], []
    for f in range(n):
        a = (w * 0.35 + 6 * np.sin(f / 6.0), h * 0.93, h * 0.75)
        people = [a]
        if f >= enter_frame:
            bx = w * 1.05 - (f - enter_frame) * (w * 0.012)
            people.append((bx, h * 0.82, h * 0.5))
        rgb, masks, _ = _render(w, h, people)
        frames.append(rgb)
        gts.append(masks)
    return frames, gts
