"""영상 변환 메인 루프 — 고정 샷 전제.

흐름:
1. 기준 프레임 → PNG → 이미지 파이프라인 Stage 0..7 → surround job 획득.
   (--surround-job 으로 기존 Job 지정 시 이 단계 건너뜀.)
2. 프레임마다: 좌/우 분리 → rectify (y 이동) → remap(미리 계산) → 블렌딩 → 인코더 파이프로.
3. 오디오는 원본에서 복사.

전제:
- 카메라 고정 (삼각대). phaseCorrelate 로 평균 이동량이 임계 초과면 경고.
- 입력 모드는 sbs (iw3 가 미리 2D → SBS 변환). mono 영상은 #3 (VDA) 에서.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vr180 import paths
from vr180.geometry import reproject
from vr180.pipeline.job import Job
from vr180.util import imageio
from vr180.util.log import RunContext
from vr180.video import frames as _frames


def _blend_weight(valid: np.ndarray, blend_px: float) -> np.ndarray:
    """valid 안쪽 경계까지 거리로 w_in = min(dist/blend_px, 1). 바깥은 0."""
    dist = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    w = np.clip(dist / max(blend_px, 1.0), 0.0, 1.0).astype(np.float32)
    w[~valid] = 0.0
    return w


def _ensure_surround_job(
    input_video: Path,
    ref_time: float,
    out_job_id: str | None,
    input_mode: str,
    ctx: RunContext,
    params_override: dict | None = None,
) -> Job:
    """기준 프레임을 PNG 로 뽑아 이미지 파이프라인을 돌린 surround job 을 돌려준다."""
    ctx.log(f"기준 프레임 추출: t={ref_time:.3f}s")
    tmp_png = paths.work_dir() / "video_refs" / f"{input_video.stem}_t{int(ref_time * 1000)}.png"
    _frames.extract_frame_png(input_video, ref_time, tmp_png)
    ctx.log(f"기준 프레임 저장: {tmp_png}")
    job = Job.create(
        tmp_png,
        job_id=out_job_id,
        params_override=params_override,
        input_mode=input_mode,
    )
    ctx.log(f"surround Job 생성: {job.id}")
    job.run(ctx=ctx)
    return job


def _check_shot_stability(
    input_path: Path,
    width: int,
    height: int,
    max_frames: int,
    threshold_px: float,
    ctx: RunContext,
) -> dict:
    """phaseCorrelate 로 프레임간 전역 이동량 평균을 측정. 임계 초과 시 경고."""
    it = _frames.frame_iter(input_path, width, height)
    prev = None
    diffs: list[float] = []
    for i, buf in enumerate(it):
        if i >= max_frames:
            break
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, gray)
            diffs.append(math.hypot(dx, dy))
        prev = gray
    mean = float(np.mean(diffs)) if diffs else 0.0
    warn = mean > threshold_px
    if warn:
        ctx.log(f"⚠️ 고정 샷이 아닐 가능성 — 프레임간 평균 이동 {mean:.2f}px > {threshold_px}px")
    else:
        ctx.log(f"샷 안정성 OK: 평균 이동 {mean:.2f}px")
    return {"mean_shift_px": mean, "warn_shot_motion": warn, "n_sampled": len(diffs)}


def _detect_shot_cuts(
    input_path: Path,
    width: int,
    height: int,
    hist_threshold: float,
    ctx: RunContext,
) -> list[int]:
    """히스토그램 chi-sq 로 급변 프레임 번호 리스트. 비싸니 저해상도로."""
    it = _frames.frame_iter(input_path, width, height)
    prev_hist = None
    cuts: list[int] = []
    for i, buf in enumerate(it):
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        small = cv2.resize(arr, (128, 72), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).flatten()
        hist = hist / (hist.sum() + 1e-9)
        if prev_hist is not None:
            d = float(cv2.compareHist(hist.astype(np.float32), prev_hist.astype(np.float32), cv2.HISTCMP_CHISQR))
            if d > hist_threshold:
                cuts.append(i)
                ctx.log(f"컷 감지: 프레임 {i} (chi-sq={d:.2f})")
        prev_hist = hist
    return cuts


def run_video(
    input_video: Path | str,
    output_video: Path | str,
    ref_frame: int | None = None,
    ref_time: float | None = None,
    surround_job: str | None = None,
    input_mode: str = "sbs",
    codec: str = "libx265",
    crf: int = 20,
    fps: float | None = None,
    check_motion: bool = True,
    motion_threshold_px: float = 2.0,
    detect_cuts: bool = False,
    cut_threshold: float = 5.0,
    params_override: dict | None = None,
    ctx: RunContext | None = None,
) -> dict:
    """영상을 VR180 SBS 영상으로 변환.

    - ref_frame 또는 ref_time 중 하나로 기준 프레임 선택 (기본: ref_frame=0).
    - surround_job 이 주어지면 그 Job 의 결과 (pano_L, pano_R, valid_mask 등) 를 재사용.
    - input_mode 는 현재 'sbs' 만 지원 (mono 는 #3 VDA).
    """
    if input_mode != "sbs":
        raise NotImplementedError(f"video input_mode={input_mode!r} 는 아직 미구현 (sbs 만 지원, mono 는 #3).")
    ctx = ctx or RunContext()
    input_video = Path(input_video)
    output_video = Path(output_video)
    info = _frames.probe(input_video)
    W, H = info["width"], info["height"]
    V_FPS = fps or info["fps"]
    ctx.log(f"입력: {input_video} {W}x{H} {V_FPS:.2f}fps  nb_frames≈{info['nb_frames']}")

    # 기준 프레임 시각
    if ref_time is None:
        ref_time = (ref_frame or 0) / max(V_FPS, 1e-6)
    ctx.log(f"기준 시각: {ref_time:.3f}s (frame≈{int(ref_time * V_FPS)})")

    # 샷 안정성/컷
    if check_motion:
        n_sample = min(info["nb_frames"] or 300, 300)
        _check_shot_stability(input_video, W, H, n_sample, motion_threshold_px, ctx)
    if detect_cuts:
        cuts = _detect_shot_cuts(input_video, W, H, cut_threshold, ctx)
        if cuts:
            ctx.log(f"⚠️ 컷 {len(cuts)}개 감지 — 1차 범위는 '고정 샷' 이라 샷마다 다른 주변부가 필요.")

    # surround job 확보
    if surround_job:
        job = Job.open(surround_job)
        ctx.log(f"surround Job 재사용: {job.id} ({job.dir})")
    else:
        job = _ensure_surround_job(input_video, ref_time, None, input_mode, ctx, params_override)

    # 고정 자산 로드
    pano_l = imageio.read_rgba(job.path(6, "pano_L.png") if input_mode == "mono" else job.path(3, "pano_L.png"))[..., :3]
    pano_r = imageio.read_rgba(job.path(6, "pano_R.png"))[..., :3]
    valid = imageio.read_mask(job.path(2, "valid_mask.png")) > 127
    cam = imageio.read_json(job.path(1, "camera.json"))
    rect = imageio.read_json(job.path(0, "rectify_report.json"))
    rect_shift = float(rect.get("applied_shift_px", 0.0) or 0.0)
    S = pano_l.shape[0]
    hfov = math.radians(float(cam["hfov_deg"]))
    ctx.log(f"surround: S={S} hfov={cam['hfov_deg']}° rect_shift={rect_shift:+.2f}px")

    # remap 테이블 (원근 → equirect). 입력은 SBS 니까 폭이 2·W_half 라고 가정.
    half_w = W // 2
    mx, my, inside = reproject.pinhole_to_equi_maps(S, half_w, H, hfov)
    # 블렌드 가중치: valid 안쪽 blend 폭 (° → px) 로 선형. Stage 6 설정 재사용.
    p6 = job.params.get("stage_06_right", {})
    blend_deg = float(p6.get("blend_deg", 8.0))
    deg_per_pixel = 180.0 / S
    blend_px = blend_deg / deg_per_pixel
    w_in = _blend_weight(valid, blend_px)

    # 인코드 파이프: 출력은 2·S × S (SBS)
    out_w = S * 2
    out_h = S

    # 프레임 처리 함수
    def process_frame(buf: bytes) -> bytes:
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
        left = arr[:, :half_w]
        right = arr[:, half_w:]
        # rectify 보정값: 첫 프레임에서 측정한 y 이동을 모든 프레임의 R 에 그대로.
        if abs(rect_shift) > 1e-3:
            m = np.float32([[1, 0, 0], [0, 1, rect_shift]])
            right = cv2.warpAffine(right, m, (half_w, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # equirect 재투영
        l_equi = cv2.remap(left, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        r_equi = cv2.remap(right, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # 블렌딩: 원본 영역 안쪽은 프레임, 바깥은 고정 pano. 경계는 w_in 로 섞음.
        w3 = w_in[..., None]
        frame_l = (w3 * l_equi.astype(np.float32) + (1 - w3) * pano_l.astype(np.float32)).astype(np.uint8)
        frame_r = (w3 * r_equi.astype(np.float32) + (1 - w3) * pano_r.astype(np.float32)).astype(np.uint8)
        sbs = np.concatenate([frame_l, frame_r], axis=1)
        return sbs.tobytes()

    # 메인 루프
    t0 = time.time()
    processed = 0
    with _frames.EncoderPipe(input_video, output_video, out_w, out_h, V_FPS, codec=codec, crf=crf) as enc:
        for i, buf in enumerate(_frames.frame_iter(input_video, W, H)):
            out_buf = process_frame(buf)
            enc.write(out_buf)
            processed += 1
            if processed % 30 == 0:
                dt = time.time() - t0
                ctx.progress(min(1.0, processed / max(info["nb_frames"], 1)), f"frame {processed}/{info['nb_frames']} ({processed / max(dt, 1e-6):.1f} fps)")
    elapsed = time.time() - t0
    ctx.log(f"완료: {processed} 프레임, {elapsed:.1f}s ({processed / max(elapsed, 1e-6):.1f} fps) → {output_video}")
    return {
        "input": str(input_video),
        "output": str(output_video),
        "surround_job": job.id,
        "frames": processed,
        "fps_out": V_FPS,
        "elapsed_s": elapsed,
        "mode": input_mode,
    }
