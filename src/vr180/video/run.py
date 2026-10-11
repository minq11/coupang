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
from vr180.geometry import reproject, warp
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
    depth_model: str = "da2_video",
    vda_chunk: int = 32,
    vda_overlap: int = 8,
    params_override: dict | None = None,
    ctx: RunContext | None = None,
) -> dict:
    """영상을 VR180 SBS 영상으로 변환.

    - ref_frame 또는 ref_time 중 하나로 기준 프레임 선택 (기본: ref_frame=0).
    - surround_job 이 주어지면 그 Job 의 결과 를 재사용.
    - input_mode='sbs' (기본): iw3 등으로 미리 SBS 변환된 영상.
    - input_mode='mono': 2D 영상 → VDA/DA2 로 프레임마다 깊이 추정 → pano_C 를 양방향 워핑.
    """
    if input_mode == "mono":
        return _run_video_mono(
            Path(input_video),
            Path(output_video),
            ref_frame=ref_frame,
            ref_time=ref_time,
            surround_job=surround_job,
            codec=codec,
            crf=crf,
            fps=fps,
            check_motion=check_motion,
            motion_threshold_px=motion_threshold_px,
            depth_model=depth_model,
            vda_chunk=vda_chunk,
            vda_overlap=vda_overlap,
            params_override=params_override,
            ctx=ctx or RunContext(),
        )
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


# ----------------------------------------------------------------------------- mono (#3)
def _normalize_pct(a: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32)
    L = float(np.percentile(aa, lo))
    H = float(np.percentile(aa, hi))
    if H - L < 1e-9:
        return np.zeros_like(aa)
    return np.clip((aa - L) / (H - L), 0.0, 1.0).astype(np.float32)


def _fit_ab_from_frame(
    invd: np.ndarray, pano_c_invd_proxy: np.ndarray | None, depth_strength: float
) -> tuple[float, float]:
    """기준 프레임의 invdepth 를 'depth_strength × normalize(invd)' 로 바로 쓰기 위한 (a, b).

    disp = a·invd + b. 가장 단순: a = depth_strength / (H − L), b = −a·L.
    프레임마다 피팅하면 떨림이 생기므로 **기준 프레임에서 1 번만** 계산.
    """
    aa = np.asarray(invd, dtype=np.float32)
    L = float(np.percentile(aa, 1))
    H = float(np.percentile(aa, 99))
    if H - L < 1e-9:
        return 0.0, 0.0
    a = float(depth_strength) / (H - L)
    b = -a * L
    return a, b


def _run_video_mono(
    input_video: Path,
    output_video: Path,
    *,
    ref_frame: int | None,
    ref_time: float | None,
    surround_job: str | None,
    codec: str,
    crf: int,
    fps: float | None,
    check_motion: bool,
    motion_threshold_px: float,
    depth_model: str,
    vda_chunk: int,
    vda_overlap: int,
    params_override: dict | None,
    ctx: RunContext,
) -> dict:
    """mono 영상 경로: 2D 입력 → 기준 프레임으로 pano_C/pano_L/pano_R 선생성 →
    프레임마다 VDA(또는 DA2) 깊이 → 가운데 뷰 ±d/2 워핑 → Telea 로 구멍 → 고정 주변부 위에 합성."""
    from vr180.models.depth_vda import get_video_depth_estimator

    info = _frames.probe(input_video)
    W, H = info["width"], info["height"]
    V_FPS = fps or info["fps"]
    ctx.log(f"[mono] 입력: {input_video} {W}x{H} {V_FPS:.2f}fps nb_frames≈{info['nb_frames']}")

    if ref_time is None:
        ref_time = (ref_frame or 0) / max(V_FPS, 1e-6)
    if check_motion:
        n = min(info["nb_frames"] or 300, 300)
        _check_shot_stability(input_video, W, H, n, motion_threshold_px, ctx)

    # 기준 프레임으로 이미지 파이프라인 (mono) 실행 → pano_C, pano_L, pano_R, valid, camera 획득
    if surround_job:
        job = Job.open(surround_job)
        ctx.log(f"[mono] surround Job 재사용: {job.id}")
    else:
        job = _ensure_surround_job(input_video, ref_time, None, "mono", ctx, params_override)

    cam = imageio.read_json(job.path(1, "camera.json"))
    pano_c = imageio.read_rgba(job.path(3, "pano_C.png"))[..., :3]
    pano_l_fixed = imageio.read_rgba(job.path(6, "pano_L.png"))[..., :3]
    pano_r_fixed = imageio.read_rgba(job.path(6, "pano_R.png"))[..., :3]
    valid = imageio.read_mask(job.path(2, "valid_mask.png")) > 127
    extent_mask = imageio.read_mask(job.path(2, "extent_mask.png")) > 127
    extent_w = imageio.read_mask(job.path(2, "extent_fade.png")).astype(np.float32) / 255.0
    S = pano_c.shape[0]
    hfov = math.radians(float(cam["hfov_deg"]))
    ctx.log(f"[mono] surround: S={S} hfov={cam['hfov_deg']}°")

    # 원근 → equirect remap 테이블. mono 영상은 전체가 센터 카메라.
    mx, my, inside = reproject.pinhole_to_equi_maps(S, W, H, hfov)

    # 블렌드 가중치 (valid 안쪽만 프레임, 바깥은 고정 pano)
    p6 = job.params.get("stage_06_right", {})
    blend_deg = float(p6.get("blend_deg", 8.0))
    deg_px = 180.0 / S
    blend_px = blend_deg / deg_px
    w_in = _blend_weight(valid, blend_px)

    # 기준 프레임 1장을 numpy 로 뽑아 a, b 눈금 1 회 피팅
    p5 = job.params.get("stage_05_depth", {})
    depth_strength = float(p5.get("depth_strength", 0.02))
    est = get_video_depth_estimator(depth_model, job.model_params(), chunk=vda_chunk, overlap=vda_overlap)
    if est is None:
        raise ValueError("mono 영상에는 depth_model 이 필요합니다 (da2_video 또는 vda_large).")

    # 기준 프레임 하나로 (a, b) 결정. VDA 는 영상 안에서 눈금이 일정하므로 1회면 충분.
    ref_png_path = paths.work_dir() / "video_refs" / f"{input_video.stem}_t{int(ref_time * 1000)}.png"
    if not ref_png_path.exists():
        _frames.extract_frame_png(input_video, ref_time, ref_png_path)
    ref_rgb = imageio.read_rgb(ref_png_path)
    ctx.log(f"[mono] 기준 프레임 깊이 추정 → a, b 1회 피팅 (depth_model={depth_model})")
    with est:
        ref_invd = est.inverse_depth_sequence([ref_rgb])[0]
    a_fit, b_fit = _fit_ab_from_frame(ref_invd, None, depth_strength)
    ctx.log(f"[mono] a={a_fit:.4f} b={b_fit:.4f} depth_strength={depth_strength:.4f}")

    # 메인 프레임 루프 — VDA 는 묶음 처리가 이상적이지만, DA2 폴백은 프레임별이라 공통 코드로.
    out_w, out_h = S * 2, S
    t0 = time.time()
    processed = 0
    with est:
        with _frames.EncoderPipe(input_video, output_video, out_w, out_h, V_FPS, codec=codec, crf=crf) as enc:
            buf_frames: list[np.ndarray] = []
            for i, buf in enumerate(_frames.frame_iter(input_video, W, H)):
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
                buf_frames.append(arr)
                # 묶음 깊이 추정 (VDA 는 chunk, DA2 폴백은 그냥 1장씩)
                if len(buf_frames) >= max(vda_chunk, 1):
                    invd_list = est.inverse_depth_sequence(buf_frames)
                    for arr_i, invd in zip(buf_frames, invd_list, strict=True):
                        _emit_mono_frame(
                            arr_i, invd, a_fit, b_fit, depth_strength,
                            mx, my, w_in, pano_l_fixed, pano_r_fixed, pano_c,
                            extent_mask, extent_w, S, enc,
                        )
                    processed += len(buf_frames)
                    buf_frames = []
                    dt = time.time() - t0
                    ctx.progress(
                        min(1.0, processed / max(info["nb_frames"], 1)),
                        f"frame {processed}/{info['nb_frames']} ({processed / max(dt, 1e-6):.1f} fps)",
                    )
            # 남은 꼬리
            if buf_frames:
                invd_list = est.inverse_depth_sequence(buf_frames)
                for arr_i, invd in zip(buf_frames, invd_list, strict=True):
                    _emit_mono_frame(
                        arr_i, invd, a_fit, b_fit, depth_strength,
                        mx, my, w_in, pano_l_fixed, pano_r_fixed, pano_c,
                        extent_mask, extent_w, S, enc,
                    )
                processed += len(buf_frames)
    elapsed = time.time() - t0
    ctx.log(f"[mono] 완료: {processed} 프레임, {elapsed:.1f}s → {output_video}")
    return {
        "input": str(input_video),
        "output": str(output_video),
        "surround_job": job.id,
        "frames": processed,
        "fps_out": V_FPS,
        "elapsed_s": elapsed,
        "mode": "mono",
        "depth_model": est.info().name,
        "a": a_fit,
        "b": b_fit,
    }


def _emit_mono_frame(
    frame_rgb: np.ndarray,
    inv_depth: np.ndarray,
    a: float,
    b: float,
    depth_strength: float,
    mx: np.ndarray,
    my: np.ndarray,
    w_in: np.ndarray,
    pano_l_fixed: np.ndarray,
    pano_r_fixed: np.ndarray,
    pano_c_fixed: np.ndarray,
    extent_mask: np.ndarray,
    extent_w: np.ndarray,
    S: int,
    enc,
) -> None:
    """프레임 1장: 원근 → equirect → ±d/2 워핑 → 구멍 Telea → 고정 pano 와 블렌딩 → 인코더."""
    # 1) 가운데 프레임을 equirect 로
    c_equi = cv2.remap(frame_rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 2) 프레임 깊이 → disparity (a, b 는 기준 프레임에서 1회만 피팅됨)
    disp = (a * inv_depth + b).astype(np.float32)
    disp = np.clip(disp, 0.0, depth_strength).astype(np.float32)
    # inv_depth 는 원근 좌표계 (H×W). disp 를 equirect 좌표로 remap.
    disp_equi = cv2.remap(disp, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    disp_equi = disp_equi * extent_w  # 범위 밖은 0 (양 눈 동일)

    # 3) 가운데 뷰를 ±d/2 로 양방향 워핑
    warped_l, cov_l = warp.forward_warp(c_equi, disp_equi, src_mask=extent_mask, taps=2, scale=+0.5)
    warped_r, cov_r = warp.forward_warp(c_equi, disp_equi, src_mask=extent_mask, taps=2, scale=-0.5)
    # 구멍은 Telea 만 (프레임마다 inpaint 모델은 금지, 비용 문제)
    holes_l = (~cov_l & extent_mask).astype(np.uint8) * 255
    holes_r = (~cov_r & extent_mask).astype(np.uint8) * 255
    if holes_l.any():
        warped_l = cv2.inpaint(warped_l, holes_l, 3, cv2.INPAINT_TELEA)
    if holes_r.any():
        warped_r = cv2.inpaint(warped_r, holes_r, 3, cv2.INPAINT_TELEA)

    # 4) 고정 주변부 위에 합성: valid 안쪽은 프레임, 바깥은 고정 pano_L/pano_R
    w3 = w_in[..., None]
    frame_l_out = (w3 * warped_l.astype(np.float32) + (1 - w3) * pano_l_fixed.astype(np.float32)).astype(np.uint8)
    frame_r_out = (w3 * warped_r.astype(np.float32) + (1 - w3) * pano_r_fixed.astype(np.float32)).astype(np.uint8)
    sbs = np.concatenate([frame_l_out, frame_r_out], axis=1)
    enc.write(sbs.tobytes())
