"""사람 추적 — "영상에 나오는 모든 사람" 을 프레임마다 한 장의 마스크(합집합)로 만든다.

처리 순서 (두 단계)
  1) 키프레임(기본 15프레임마다)에서 사람 검출 → 키프레임끼리 이어 붙여 '트랙' 목록을 만든다.
     - 박스를 25% 넓힌 IoU 가 iou_new 이상이거나, 박스 중심이 몸 높이 × link_dist 이내로 움직였고 크기가
       비슷하면 같은 사람 (걷는 사람은 0.5초에 몸 폭보다 많이 움직여 박스가 안 겹친다). 아니면 새 사람
     - 키프레임 min_confirm 번 이상 잡히거나 점수가 strong_score 이상이어야 등록 (포스터·마네킹 오검출 걸러내기)
     - max_miss 키프레임 연속으로 안 잡히면 그 뒤로는 끝 (화면 밖으로 나감 / 추적 표류 방지)
     - 너무 작은 사람(min_area_frac 미만)은 무시, 등록 인원은 max_people 까지
  2) SAM 2.1 에 각 트랙의 첫 키프레임 마스크를 한꺼번에 넣고, 앞으로 한 번 · 뒤로 한 번 전파.
     첫 키프레임 이후는 앞 방향 결과, 그 직전 구간(최대 interval 프레임)은 뒤 방향 결과를 쓴다
     → 키프레임 사이에 들어온 사람도 들어오던 순간부터 잡힌다.
"모든 사람" 이 목표라 같은 사람이 두 트랙으로 잡혀도 합집합 결과는 같다 (계산만 조금 늘어남).

SAM 2 가 없으면 track_detect_each_frame: 매 프레임 검출 + 앞뒤 3프레임 다수결 (깜빡임은 더 있음).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np

from vr180.person.models import Detection, PersonDetector

# ----------------------------------------------------------------------------- 결과 자료형


@dataclass
class Track:
    tid: int
    first_frame: int
    prompt_mask: np.ndarray
    last_box: tuple[int, int, int, int]
    hits: list[int] = field(default_factory=list)
    best_score: float = 0.0
    misses: int = 0
    end_frame: int | None = None

    def summary(self) -> dict:
        return {
            "id": self.tid,
            "first_frame": self.first_frame,
            "end_frame": self.end_frame,
            "keyframe_hits": len(self.hits),
            "best_score": round(self.best_score, 3),
        }


@dataclass
class TrackResult:
    height: int
    width: int
    masks: list[np.ndarray]  # 프레임별 합집합 마스크 (np.packbits 로 압축)
    people: list[dict]
    tallest_frac: float = 0.0  # 가장 크게 나온 사람의 키 / 화면 높이
    tallest_bottom: float = 1.0  # 그 사람 발끝 위치 (위=0, 아래=1)

    def mask(self, i: int) -> np.ndarray:
        n = self.height * self.width
        return np.unpackbits(self.masks[i], count=n).reshape(self.height, self.width).astype(bool)

    def __len__(self) -> int:
        return len(self.masks)


def _pack(m: np.ndarray) -> np.ndarray:
    return np.packbits(m.astype(bool).ravel())


class _Stats:
    """가장 크게 나온 사람 (플레이어가 판넬 크기를 '키 1.7m' 로 맞출 때 기준)."""

    def __init__(self, h: int):
        self.h = h
        self.frac = 0.0
        self.bottom = 1.0

    def update(self, m: np.ndarray) -> None:
        rows = np.flatnonzero(m.any(axis=1))
        if rows.size == 0:
            return
        frac = (rows[-1] + 1 - rows[0]) / self.h
        if frac > self.frac:
            self.frac = float(frac)
            self.bottom = float((rows[-1] + 1) / self.h)


# ----------------------------------------------------------------------------- 1) 키프레임 연결


def _expand_box(b, e: float):
    x0, y0, x1, y1 = b
    dx, dy = (x1 - x0) * e, (y1 - y0) * e
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def box_iou(a, b, expand: float = 0.25) -> float:
    a, b = _expand_box(a, expand), _expand_box(b, expand)
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def link_score(a, b, iou_new: float, link_dist: float) -> float:
    """같은 사람일 가능성. iou_new 이상이면 박스 겹침으로 확정, 그 아래 (0, iou_new) 는 중심 거리로 이어진 경우."""
    v = box_iou(a, b)
    if v >= iou_new:
        return v
    ha, hb = a[3] - a[1], b[3] - b[1]
    if min(ha, hb) <= 0 or max(ha, hb) / min(ha, hb) > 2.0:
        return 0.0
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    lim = link_dist * max(ha, hb)
    if dist > lim:
        return 0.0
    return max(1e-3, iou_new * 0.99 * (1.0 - dist / lim))


def link_keyframes(
    kf_dets: list[tuple[int, list[Detection]]],
    *,
    n_frames: int,
    interval: int,
    frame_area: int,
    iou_new: float = 0.3,
    link_dist: float = 1.0,
    min_confirm: int = 2,
    strong_score: float = 0.8,
    min_area_frac: float = 0.002,
    max_miss: int = 3,
    max_people: int = 8,
) -> list[Track]:
    tracks: list[Track] = []
    active: list[Track] = []
    for f, dets in kf_dets:
        dets = [d for d in dets if d.area >= min_area_frac * frame_area]
        pairs = sorted(
            (
                (link_score(t.last_box, d.box, iou_new, link_dist), ti, di)
                for ti, t in enumerate(active)
                for di, d in enumerate(dets)
            ),
            reverse=True,
        )
        used_t: set[int] = set()
        used_d: set[int] = set()
        for v, ti, di in pairs:
            if v <= 0:
                break
            if ti in used_t or di in used_d:
                continue
            t, d = active[ti], dets[di]
            t.hits.append(f)
            t.last_box = d.box
            t.misses = 0
            t.best_score = max(t.best_score, d.score)
            used_t.add(ti)
            used_d.add(di)
        for ti, t in enumerate(active):
            if ti in used_t:
                continue
            t.misses += 1
            if t.misses >= max_miss:
                t.end_frame = min(n_frames - 1, t.hits[-1] + interval)
        active = [t for t in active if t.end_frame is None]
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            t = Track(tid=0, first_frame=f, prompt_mask=d.mask, last_box=d.box, hits=[f], best_score=d.score)
            tracks.append(t)
            active.append(t)

    single = len(kf_dets) <= 1
    keep = [t for t in tracks if single or len(t.hits) >= min_confirm or t.best_score >= strong_score]
    keep.sort(key=lambda t: (-len(t.hits), -t.best_score))
    keep = keep[: max(1, int(max_people))]
    keep.sort(key=lambda t: t.first_frame)
    for i, t in enumerate(keep, start=1):
        t.tid = i
    return keep


def detect_keyframes(
    detector: PersonDetector,
    load_frame: Callable[[int], np.ndarray],
    n_frames: int,
    interval: int,
    ctx=None,
) -> list[tuple[int, list[Detection]]]:
    idx = list(range(0, n_frames, max(1, interval)))
    if idx[-1] != n_frames - 1 and n_frames - 1 - idx[-1] > interval // 2:
        idx.append(n_frames - 1)  # 끝에 들어온 사람도 잡히게
    out = []
    for k, f in enumerate(idx):
        out.append((f, detector.detect(load_frame(f))))
        if ctx is not None:
            ctx.progress(k / max(len(idx), 1), f"사람 찾기 {k + 1}/{len(idx)}")
    return out


# ----------------------------------------------------------------------------- 2) SAM 2.1 전파


def track_sam2(
    predictor,
    frames_dir: str,
    tracks: list[Track],
    *,
    n_frames: int,
    height: int,
    width: int,
    interval: int,
    offload_state: bool = False,
    ctx=None,
) -> TrackResult:
    """predictor: SAM2VideoPredictor (또는 같은 메서드를 가진 테스트용 가짜)."""
    union = [np.zeros((height, width), dtype=bool) for _ in range(n_frames)]
    stats = _Stats(height)
    if not tracks:
        return TrackResult(height, width, [_pack(m) for m in union], [], 0.0, 1.0)

    first = {t.tid: t.first_frame for t in tracks}
    end = {t.tid: (t.end_frame if t.end_frame is not None else n_frames - 1) for t in tracks}

    def accumulate(f: int, obj_ids, logits, valid) -> None:
        masks = _to_numpy_masks(logits)
        for k, oid in enumerate(obj_ids):
            if valid(int(oid), f) and masks[k].any():
                union[f] |= masks[k]
                stats.update(masks[k])

    try:
        import torch

        cuda = torch.cuda.is_available()
        guard = torch.inference_mode()
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else nullcontext()
    except ImportError:  # 테스트용 가짜 predictor
        guard, amp = nullcontext(), nullcontext()

    with guard, amp:
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=bool(offload_state),
            async_loading_frames=False,
        )
        for t in tracks:
            predictor.add_new_mask(state, frame_idx=t.first_frame, obj_id=t.tid, mask=t.prompt_mask)

        start = min(first.values())
        total = (n_frames - start) + max(first.values())
        done = 0
        # 앞 방향: 그 사람의 첫 키프레임 ~ 끝 프레임까지만 인정
        for f, obj_ids, logits in predictor.propagate_in_video(state):
            accumulate(f, obj_ids, logits, lambda oid, fr: first[oid] <= fr <= end[oid])
            done += 1
            if ctx is not None:
                ctx.progress(min(1.0, done / max(total, 1)), f"따라가기 (앞) {f + 1}/{n_frames}")
        # 뒤 방향: 첫 키프레임 직전 구간 (키프레임 사이에 들어온 사람을 들어온 순간부터)
        back_start = max(first.values())
        if back_start > 0:
            for f, obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=back_start, reverse=True
            ):
                accumulate(f, obj_ids, logits, lambda oid, fr: first[oid] - interval <= fr < first[oid])
                done += 1
                if ctx is not None:
                    ctx.progress(min(1.0, done / max(total, 1)), f"따라가기 (뒤) {f + 1}/{n_frames}")
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(state)

    return TrackResult(
        height, width, [_pack(m) for m in union], [t.summary() for t in tracks], stats.frac, stats.bottom
    )


def _to_numpy_masks(logits) -> np.ndarray:
    """SAM 2 출력 (N,1,H,W) logits (torch, GPU 일 수 있음) → bool (N,H,W). 바로 CPU 로 내려 GPU 메모리를 안 쌓는다."""
    if hasattr(logits, "detach"):
        return (logits[:, 0] > 0).detach().cpu().numpy()
    arr = np.asarray(logits)
    return arr[:, 0] > 0 if arr.ndim == 4 else arr > 0


# ----------------------------------------------------------------------------- 대안: 매 프레임 검출


def track_detect_each_frame(
    detector: PersonDetector,
    load_frame: Callable[[int], np.ndarray],
    n_frames: int,
    *,
    min_area_frac: float = 0.002,
    ctx=None,
) -> TrackResult:
    frames_masks: list[np.ndarray] = []
    h = w = None
    for f in range(n_frames):
        rgb = load_frame(f)
        h, w = rgb.shape[:2]
        m = np.zeros((h, w), dtype=bool)
        for d in detector.detect(rgb):
            if d.area >= min_area_frac * h * w:
                m |= d.mask
        frames_masks.append(m)
        if ctx is not None:
            ctx.progress(f / max(n_frames, 1), f"사람 찾기 (매 프레임) {f + 1}/{n_frames}")
    stats = _Stats(h)
    out = []
    for f in range(n_frames):  # 앞뒤 3프레임 다수결 → 한 프레임짜리 깜빡임 제거
        a = frames_masks[max(0, f - 1)]
        b = frames_masks[f]
        c = frames_masks[min(n_frames - 1, f + 1)]
        m = (a.astype(np.uint8) + b + c) >= 2
        stats.update(m)
        out.append(_pack(m))
    return TrackResult(h, w, out, [], stats.frac, stats.bottom)


def single_image_masks(
    detector: PersonDetector, rgb: np.ndarray, *, min_area_frac: float = 0.002
) -> TrackResult:
    h, w = rgb.shape[:2]
    m = np.zeros((h, w), dtype=bool)
    people = []
    stats = _Stats(h)
    for i, d in enumerate(detector.detect(rgb), start=1):
        if d.area < min_area_frac * h * w:
            continue
        m |= d.mask
        stats.update(d.mask)
        people.append({"id": i, "score": round(d.score, 3), "box": list(d.box)})
    return TrackResult(h, w, [_pack(m)], people, stats.frac, stats.bottom)
