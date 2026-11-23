"""인물 AR 파이프라인 — 사진·영상 공통 진입점 run_person().

작업 폴더 work/person/<id>/ 에 단계별 중간 파일을 남긴다 (열어 보고, 고치고, 그 단계부터 다시 돌릴 수 있게).
  00_frames/   입력을 처리 해상도로 줄인 프레임 (영상은 00000.jpg …, SAM 2 입력 형식)
  01_masks/    프레임별 사람 합집합 마스크 PNG + people.json (트랙 목록, 가장 큰 사람 키)
  02_alpha/    프레임별 알파 PNG (경계 다듬기 + 경계 띠 앞뒤 평균)
  03_depth/    프레임별 inverse depth (float16 .npy, 저장용 축소) + stats.json
  out/         {원본}_{id}_personar_rgba.(png|mp4)  위 컬러 / 아래 알파, 눈당 좌우 (layout rgba_sbs_v1, 우리 플레이어 전용)
               {원본}_{id}_LRF_Full_SBS.(png|mp4)   초록 배경 SBS (Skybox·DeoVR 등이 파일명으로 3D 좌우 인식)
               이름 규칙은 volumx 와 같다: ASCII 로 정리한 원본 이름 + 32자 hex id + 형식 태그 (output_names).
               poster.png, preview.png, media.json (플레이어가 읽는 설명서)
단계마다 meta.json 에 입력·파라미터 지문을 적어 두고, 같으면 건너뛴다. 사용자가 01_masks / 02_alpha 의 PNG 를
직접 고쳐 덮어써도 다음 단계는 그 파일을 그대로 읽는다.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vr180 import paths
from vr180.person import matte, stereo
from vr180.person import track as T
from vr180.person.models import (
    MaskFileDetector,
    get_matter,
    get_person_detector,
    load_sam2_video_predictor,
    sam2_available,
)
from vr180.util import imageio
from vr180.util.hashing import params_hash
from vr180.util.log import RunContext

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULTS: dict[str, Any] = {
    "max_eye_width": 1920,
    "detector": "rfdetr_seg_medium",
    "detect_threshold": 0.4,
    "detect_long_side": 1032,
    "min_area_frac": 0.002,
    "keyframe_sec": 0.5,
    "iou_new": 0.3,
    "link_dist": 1.0,
    "min_confirm": 2,
    "strong_score": 0.8,
    "max_miss": 3,
    "max_people": 8,
    "tracker": "sam2",
    "sam2_size": "small",
    "sam2_offload_state": False,
    "sam2_segment": 300,
    "matter": "birefnet_matting",
    "core_frac": 0.010,
    "band_frac": 0.016,
    "image_depth_model": "da2_small",
    "video_depth_model": "vda_small",
    "depth_group": 240,
    "depth_overlap": 16,
    "depth_store_width": 960,
    "relief_px": 10.0,
    "convergence": "front",
    "stereo_workers": 0,
    "codec": "libx265",
    "crf": 18,
    "mask_path": None,
}


def person_params(override: dict | None = None) -> dict:
    from vr180.pipeline.job import _deep_update, load_default_params

    p = copy.deepcopy(DEFAULTS)
    try:
        p.update(load_default_params().get("person", {}) or {})
    except FileNotFoundError:
        pass
    if override:
        _deep_update(p, override)
    return p


def models_cfg() -> dict:
    from vr180.pipeline.job import load_default_params

    return dict(load_default_params().get("models", {}))


@dataclass
class PersonModels:
    """테스트·고급 사용: 직접 만든 모델을 넣으면 그걸 쓴다 (None 이면 파라미터로 만든다)."""

    detector: Any = None
    matter: Any = None
    depth: Any = (
        None  # 사진: DepthEstimator(inverse_depth) / 영상: VideoDepthEstimator(inverse_depth_sequence)
    )
    sam2: Any = None  # SAM2VideoPredictor 와 같은 메서드를 가진 객체
    use_matter: bool = True  # False 면 matter 를 만들지 않고 모델 없는 부드럽게 처리
    use_depth: bool = True


def processing_size(w: int, h: int, max_eye_width: int) -> tuple[int, int]:
    s = min(1.0, float(max_eye_width) / float(w))
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


def _file_sig(path: Path) -> str:
    st = path.stat()
    return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"


class _Step:
    def __init__(self, d: Path, key_parts: dict):
        self.dir = d
        self.key = params_hash(key_parts)
        self.meta_path = d / "meta.json"

    def fresh(self) -> bool:
        if not self.meta_path.exists():
            return False
        m = imageio.read_json(self.meta_path)
        return m.get("key") == self.key and m.get("done", False)

    def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.meta_path.exists():
            self.meta_path.unlink()

    def finish(self, extra: dict | None = None) -> None:
        imageio.write_json(
            self.meta_path,
            {
                "key": self.key,
                "done": True,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                **(extra or {}),
            },
        )

    def meta(self) -> dict:
        return imageio.read_json(self.meta_path)


def _dir_sig(d: Path, pattern: str = "*.png") -> str:
    """폴더 안 파일들이 바뀌었는지 (사용자가 마스크를 고쳐 덮어쓴 경우 다음 단계를 다시 돌리기 위해)."""
    fs = list(d.glob(pattern))
    if not fs:
        return "empty"
    st = [f.stat() for f in fs]
    return f"{len(fs)}:{max(x.st_mtime_ns for x in st)}:{sum(x.st_size for x in st)}"


def _px(frac: float, h: int) -> int:
    return max(1, int(round(frac * h)))


def _relief(p: dict, h: int) -> float:
    return float(p["relief_px"]) * h / 1080.0


def _t0(st: dict, p: dict) -> float:
    return st["t_front"] if str(p.get("convergence", "front")) == "front" else st["t_mean"]


def _unload(m) -> None:
    if m is None:
        return
    for name in ("unload", "_unload"):
        fn = getattr(m, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
            break
    from vr180.models.base import free_cuda

    free_cuda()


# ============================================================================= 진입점


def run_person(
    input_path: str | Path,
    *,
    job_dir: str | Path | None = None,
    params: dict | None = None,
    models: PersonModels | None = None,
    ctx: RunContext | None = None,
) -> dict:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    ctx = ctx or RunContext()
    p = person_params(params)
    models = models or PersonModels()
    if job_dir is None:
        job_dir = paths.work_dir() / "person" / (time.strftime("%Y%m%d_%H%M%S") + "_" + input_path.stem[:24])
    job_dir = Path(job_dir)
    (job_dir / "out").mkdir(parents=True, exist_ok=True)
    imageio.write_json(job_dir / "params.json", {"input": str(input_path.resolve()), **p})
    kind = "image" if input_path.suffix.lower() in IMAGE_EXT else "video"
    ctx.log(f"[인물AR] {kind} 입력: {input_path.name} → {job_dir}")
    t0 = time.time()
    if kind == "image":
        res = _run_image(input_path, job_dir, p, models, ctx)
    else:
        res = _run_video(input_path, job_dir, p, models, ctx)
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["job_dir"] = str(job_dir)
    ctx.log(f"[인물AR] 완료 ({res['elapsed_s']}s): {res['output']}")
    return res


# ============================================================================= 공통 모델 준비


def _detector(p: dict, models: PersonModels):
    if models.detector is not None:
        return models.detector
    if p.get("mask_path"):
        return MaskFileDetector({}, p["mask_path"])
    return get_person_detector(
        p["detector"],
        models_cfg(),
        threshold=float(p["detect_threshold"]),
        long_side=int(p["detect_long_side"]),
    )


def _matter(p: dict, models: PersonModels):
    if not models.use_matter:
        return None
    if models.matter is not None:
        return models.matter
    return get_matter(p["matter"], models_cfg())


# 형식 태그. 플레이어들은 파일명 토큰으로 배치를 판단한다 (volumx 의 iw3 결과도 _LRF_Full_SBS).
# rgba 는 우리 플레이어 전용 배치라 SBS / LR / TB 토큰을 일부러 넣지 않는다 (일반 플레이어가 3D 로 오해).
TAG_GREEN = "LRF_Full_SBS"
TAG_PACKED = "personar_rgba"
_LEGACY_NAMES = ("person_rgba_sbs", "person_green_sbs")


def sanitize_filename(name: str) -> str:
    """volumx(backend/app/utils/s3.py) 와 같은 규칙: ASCII 영숫자 · - _ . 만 남기고 나머지는 _, 연속 _ 하나로, 앞뒤 _ . 제거."""
    safe = "".join(c if ord(c) < 128 and (c.isalnum() or c in "-_.") else "_" for c in name)
    safe = re.sub(r"_+", "_", safe).strip("_.")
    return safe or "file"


def output_names(src: Path, job_dir: Path, ext: str) -> tuple[str, str]:
    """(rgba 파일명, 초록 SBS 파일명). volumx 처럼 {원본}_{32자 hex}_{태그}.{ext}.
    id 는 Job 폴더 이름에서 만들어(uuid5) 같은 Job 을 다시 돌려도 이름이 바뀌지 않는다."""
    base = sanitize_filename(Path(src).stem)
    uid = uuid.uuid5(uuid.NAMESPACE_URL, f"vr180-person/{Path(job_dir).name}").hex
    return f"{base}_{uid}_{TAG_PACKED}.{ext}", f"{base}_{uid}_{TAG_GREEN}.{ext}"


def _clean_old_outputs(out: Path, keep: tuple[str, ...]) -> None:
    """이름 규칙이 바뀌기 전 파일(person_rgba_sbs.*) 이나 원본 이름이 달랐던 이전 결과를 지운다."""
    for f in out.iterdir():
        if f.name in keep or not f.is_file():
            continue
        if f.stem in _LEGACY_NAMES or f.stem.endswith(("_" + TAG_PACKED, "_" + TAG_GREEN)):
            f.unlink(missing_ok=True)


def _write_media(job_dir: Path, info: dict) -> None:
    imageio.write_json(
        job_dir / "out" / "media.json",
        {"layout": stereo.LAYOUT, "created": time.strftime("%Y-%m-%d %H:%M:%S"), **info},
    )


# ============================================================================= 사진


def _run_image(src: Path, job_dir: Path, p: dict, models: PersonModels, ctx: RunContext) -> dict:
    rgb0 = imageio.read_rgb(src)
    W, H = processing_size(rgb0.shape[1], rgb0.shape[0], int(p["max_eye_width"]))
    rgb = (
        cv2.resize(rgb0, (W, H), interpolation=cv2.INTER_AREA)
        if (W, H) != (rgb0.shape[1], rgb0.shape[0])
        else rgb0
    )
    d0, d1, d2, d3, out = (job_dir / n for n in ("00_frames", "01_masks", "02_alpha", "03_depth", "out"))
    imageio.write_image(d0 / "frame.png", rgb)
    sig = _file_sig(src)

    # 1 사람 찾기
    s1 = _Step(
        d1,
        {
            "src": sig,
            "size": [W, H],
            "det": [
                p["detector"],
                p["detect_threshold"],
                p["detect_long_side"],
                p["min_area_frac"],
                p.get("mask_path"),
            ],
        },
    )
    if not s1.fresh() or models.detector is not None:
        s1.start()
        ctx.log("1/4 사람 찾기")
        det = _detector(p, models)
        try:
            det.load()
            tr = T.single_image_masks(det, rgb, min_area_frac=float(p["min_area_frac"]))
        finally:
            if models.detector is None:
                _unload(det)
        imageio.write_image(d1 / "mask.png", tr.mask(0).astype(np.uint8) * 255)
        imageio.write_json(
            d1 / "people.json",
            {"people": tr.people, "tallest_frac": tr.tallest_frac, "tallest_bottom": tr.tallest_bottom},
        )
        s1.finish({"people": len(tr.people)})
    mask = imageio.read_mask(d1 / "mask.png") > 127
    people = imageio.read_json(d1 / "people.json")
    if not mask.any():
        raise RuntimeError(
            "사진에서 사람을 찾지 못했습니다 (detect_threshold 를 낮추거나 mask_path 로 직접 지정)."
        )
    ctx.log(f"  사람 {len(people['people'])}명")

    # 2 경계 다듬기
    core_px, band_px = _px(p["core_frac"], H), _px(p["band_frac"], H)
    s2 = _Step(
        d2,
        {
            "s1": s1.key,
            "mask_mtime": (d1 / "mask.png").stat().st_mtime_ns,
            "matter": p["matter"] if models.use_matter else "none",
            "core": core_px,
            "band": band_px,
        },
    )
    if not s2.fresh() or models.matter is not None:
        s2.start()
        ctx.log("2/4 경계 다듬기")
        mt = _matter(p, models)
        try:
            alpha = matte.refine_alpha(rgb, mask, mt, core_px=core_px, band_px=band_px)
        finally:
            if models.matter is None:
                _unload(mt)
        imageio.write_image(d2 / "alpha.png", (alpha * 255 + 0.5).astype(np.uint8))
        s2.finish()
    alpha_u8 = imageio.read_mask(d2 / "alpha.png")

    # 3 깊이 (사람만 잘라 넣지 않고 전체 프레임으로)
    s3 = _Step(
        d3, {"src": sig, "size": [W, H], "model": p["image_depth_model"] if models.use_depth else "none"}
    )
    if not s3.fresh() or models.depth is not None:
        s3.start()
        ctx.log(f"3/4 깊이 ({p['image_depth_model'] if models.use_depth else 'none'})")
        inv = _image_depth(rgb, p, models)
        np.save(d3 / "inv_depth.npy", inv.astype(np.float32))
        prev, _ = imageio.colormap_preview(inv)
        imageio.write_image(d3 / "preview.png", prev)
        s3.finish()
    inv = np.load(d3 / "inv_depth.npy")

    # 4 좌우 영상 + 포장
    ctx.log("4/4 좌우 영상")
    st = stereo.depth_stats(inv, mask) or {"lo": 0.0, "hi": 1.0, "t_front": 1.0, "t_mean": 1.0}
    relief = _relief(p, H)
    disp = stereo.disparity_px(inv, mask, lo=st["lo"], hi=st["hi"], t0=_t0(st, p), relief_px=relief)
    left, right = stereo.eye_views(rgb, alpha_u8, disp)
    packed = stereo.pack(left, right)
    packed_name, green_name = output_names(src, job_dir, "png")
    _clean_old_outputs(out, (packed_name, green_name))
    imageio.write_image(out / packed_name, packed)
    imageio.write_image(out / green_name, stereo.green_sbs(left, right))
    imageio.write_image(out / "poster.png", packed)
    imageio.write_image(out / "preview.png", stereo.preview(left, right))
    info = {
        "type": "image",
        "file": packed_name,
        "green_file": green_name,
        "poster": "poster.png",
        "eye_width": W,
        "eye_height": H,
        "tallest_frac": people.get("tallest_frac", 0.0),
        "tallest_bottom": people.get("tallest_bottom", 1.0),
        "people": len(people["people"]),
        "relief_px": relief,
        "disp_range_px": [float(disp.min()), float(disp.max())],
    }
    _write_media(job_dir, info)
    return {
        "kind": "image",
        "output": str(out / packed_name),
        "green": str(out / green_name),
        "preview": str(out / "preview.png"),
        **info,
    }


def _image_depth(rgb: np.ndarray, p: dict, models: PersonModels) -> np.ndarray:
    h, w = rgb.shape[:2]
    if not models.use_depth:
        return np.zeros((h, w), np.float32)
    est = models.depth
    if est is None:
        from vr180.models.registry import get_depth_estimator

        est = get_depth_estimator(p["image_depth_model"], models_cfg())
        if est is None:
            return np.zeros((h, w), np.float32)
        try:
            with est:
                return est.inverse_depth(rgb).astype(np.float32)
        finally:
            _unload(est)
    return est.inverse_depth(rgb).astype(np.float32)


# ============================================================================= 영상


def _run_video(src: Path, job_dir: Path, p: dict, models: PersonModels, ctx: RunContext) -> dict:
    from vr180.video import frames as F

    info = F.probe(src)
    fps = float(info["fps"]) or 30.0
    W, H = processing_size(info["width"], info["height"], int(p["max_eye_width"]))
    d0, d1, d2, d3, out = (job_dir / n for n in ("00_frames", "01_masks", "02_alpha", "03_depth", "out"))
    sig = _file_sig(src)
    ctx.log(f"  {info['width']}x{info['height']} {fps:.2f}fps → 처리 {W}x{H}")

    # 0 프레임 뽑기
    s0 = _Step(d0, {"src": sig, "size": [W, H]})
    if not s0.fresh():
        s0.start()
        ctx.log("0/5 프레임 뽑기")
        n = F.extract_frames_jpg(src, d0, W, H)
        s0.finish({"frames": n})
    n = int(s0.meta()["frames"])
    if n == 0:
        raise RuntimeError("영상에서 프레임을 읽지 못했습니다.")

    def load_frame(i: int) -> np.ndarray:
        return imageio.read_rgb(d0 / f"{i:05d}.jpg")


    # 1 사람 찾기 + 따라가기
    interval = max(1, int(round(fps * float(p["keyframe_sec"]))))
    s1 = _Step(
        d1,
        {
            "s0": s0.key,
            "det": [p["detector"], p["detect_threshold"], p["detect_long_side"], p["min_area_frac"]],
            "link": [
                interval,
                p["iou_new"],
                p["link_dist"],
                p["min_confirm"],
                p["strong_score"],
                p["max_miss"],
                p["max_people"],
            ],
            "tracker": [p["tracker"], p["sam2_size"]],
            # 구간 나누기는 실제로 나눌 때만 키에 넣는다 (짧은 영상의 기존 캐시 유지)
            **({"segment": int(p["sam2_segment"])} if 0 < int(p.get("sam2_segment") or 0) < n else {}),
        },
    )
    if not s1.fresh() or models.detector is not None:
        s1.start()
        ctx.log(f"1/5 사람 찾기 ({interval}프레임마다) + 따라가기 ({p['tracker']})")
        tr = _track_video(load_frame, n, W, H, interval, d0, p, models, ctx)
        for i in range(n):
            imageio.write_image(d1 / f"{i:05d}.png", tr.mask(i).astype(np.uint8) * 255)
        imageio.write_json(
            d1 / "people.json",
            {
                "people": tr.people,
                "tallest_frac": tr.tallest_frac,
                "tallest_bottom": tr.tallest_bottom,
                "tracker": tr_name(tr),
            },
        )
        s1.finish({"people": len(tr.people)})
    people = imageio.read_json(d1 / "people.json")
    ctx.log(f"  등록된 사람 {len(people['people'])}명")

    def load_mask(i: int) -> np.ndarray:
        return imageio.read_mask(d1 / f"{i:05d}.png") > 127

    # 2 경계 다듬기 (+ 경계 띠 앞뒤 평균)
    core_px, band_px = _px(p["core_frac"], H), _px(p["band_frac"], H)
    masks_sig = _dir_sig(d1)
    s2 = _Step(
        d2,
        {
            "s1": s1.key,
            "masks": masks_sig,
            "matter": p["matter"] if models.use_matter else "none",
            "core": core_px,
            "band": band_px,
        },
    )
    if not s2.fresh() or models.matter is not None:
        s2.start()
        ctx.log("2/5 경계 다듬기")
        raw = d2 / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        mt = _matter(p, models)
        try:
            for i in range(n):
                a = matte.refine_alpha(load_frame(i), load_mask(i), mt, core_px=core_px, band_px=band_px)
                imageio.write_image(raw / f"{i:05d}.png", (a * 255 + 0.5).astype(np.uint8))
                ctx.progress((i + 1) / n, f"경계 {i + 1}/{n}")
        finally:
            if models.matter is None:
                _unload(mt)
        prev = None
        cur = imageio.read_mask(raw / "00000.png")
        for i in range(n):
            nxt = imageio.read_mask(raw / f"{i + 1:05d}.png") if i + 1 < n else None
            band = matte.band_of(load_mask(i), core_px, band_px)
            imageio.write_image(d2 / f"{i:05d}.png", matte.temporal_smooth(prev, cur, nxt, band))
            prev, cur = cur, nxt
        shutil.rmtree(raw, ignore_errors=True)
        s2.finish()

    # 3 깊이
    s3 = _Step(
        d3,
        {
            "s0": s0.key,
            "s1": s1.key,
            "masks": masks_sig,
            "model": p["video_depth_model"] if models.use_depth else "none",
            "group": [p["depth_group"], p["depth_overlap"]],
            "store": p["depth_store_width"],
        },
    )
    if not s3.fresh() or models.depth is not None:
        s3.start()
        ctx.log(f"3/5 깊이 ({p['video_depth_model'] if models.use_depth else 'none'})")
        stats = _video_depth(load_frame, load_mask, n, W, H, d3, p, models, ctx)
        imageio.write_json(d3 / "stats.json", stats)
        s3.finish()
    stats = imageio.read_json(d3 / "stats.json")

    # 4·5 좌우 영상 + 인코딩
    ctx.log("4/5 좌우 영상 · 5/5 인코딩")
    relief = _relief(p, H)
    lo, hi, t0 = (
        stats["lo"],
        stats["hi"],
        stats["t_front"] if p["convergence"] == "front" else stats["t_mean"],
    )
    packed_name, green_name = output_names(src, job_dir, "mp4")
    _clean_old_outputs(out, (packed_name, green_name))
    packed_path = out / packed_name
    green_path = out / green_name
    dmin, dmax = 0.0, 0.0
    with (
        F.EncoderPipe(src, packed_path, 2 * W, 2 * H, fps, codec=p["codec"], crf=int(p["crf"])) as enc_a,
        F.EncoderPipe(src, green_path, 2 * W, H, fps, codec=p["codec"], crf=int(p["crf"])) as enc_g,
    ):

        def make(i: int):
            mask = load_mask(i)
            alpha_u8 = imageio.read_mask(d2 / f"{i:05d}.png")
            inv = np.load(d3 / f"{i:05d}.npy").astype(np.float32)
            disp = stereo.disparity_px(inv, mask, lo=lo, hi=hi, t0=t0, relief_px=relief)
            left, right = stereo.eye_views(load_frame(i), alpha_u8, disp)
            rng = (float(disp[mask].min()), float(disp[mask].max())) if mask.any() else None
            prev = stereo.preview(left, right) if i == 0 else None
            return stereo.pack(left, right), stereo.green_sbs(left, right), rng, prev

        # 프레임마다 독립 (워핑·포장 ≈ 0.65s/프레임, 1080x1920). cv2·numpy 가 GIL 을 풀어 스레드로 나눈다.
        # 인코더 두 개(x265)도 CPU 를 쓰므로 코어의 절반만. 순서는 제출 순서대로 꺼내 유지.
        workers = max(1, min(int(p.get("stereo_workers") or 0) or (os.cpu_count() or 2) // 2, 8))
        with ThreadPoolExecutor(workers) as pool:
            pending = [pool.submit(make, i) for i in range(min(n, 2 * workers))]
            for i in range(n):
                packed, green, rng, prev = pending.pop(0).result()
                if i + len(pending) + 1 < n:
                    pending.append(pool.submit(make, i + len(pending) + 1))
                enc_a.write(packed.tobytes())
                enc_g.write(green.tobytes())
                if rng:
                    dmin, dmax = min(dmin, rng[0]), max(dmax, rng[1])
                if i == 0:
                    imageio.write_image(out / "poster.png", packed)
                    imageio.write_image(out / "preview.png", prev)
                ctx.progress((i + 1) / n, f"인코딩 {i + 1}/{n}")
    media = {
        "type": "video",
        "file": packed_path.name,
        "green_file": green_path.name,
        "poster": "poster.png",
        "eye_width": W,
        "eye_height": H,
        "fps": fps,
        "frames": n,
        "tallest_frac": people.get("tallest_frac", 0.0),
        "tallest_bottom": people.get("tallest_bottom", 1.0),
        "people": len(people["people"]),
        "relief_px": relief,
        "disp_range_px": [dmin, dmax],
    }
    _write_media(job_dir, media)
    return {
        "kind": "video",
        "output": str(packed_path),
        "green": str(green_path),
        "preview": str(out / "preview.png"),
        **media,
    }


def tr_name(tr: T.TrackResult) -> str:
    return "sam2" if tr.people and "first_frame" in tr.people[0] else "frame_detect"


def _track_video(load_frame, n, W, H, interval, frames_dir, p, models: PersonModels, ctx) -> T.TrackResult:
    det = _detector(p, models)
    use_sam2 = p["tracker"] == "sam2"
    reason = ""
    if use_sam2 and models.sam2 is None:
        ok, reason = sam2_available(p["sam2_size"])
        use_sam2 = ok
    try:
        det.load()
        if not use_sam2:
            if p["tracker"] == "sam2":
                ctx.log(f"  SAM 2 를 쓸 수 없어 매 프레임 검출로 대신합니다: {reason}")
            return T.track_detect_each_frame(
                det, load_frame, n, min_area_frac=float(p["min_area_frac"]), ctx=ctx
            )
        kf = T.detect_keyframes(det, load_frame, n, interval, ctx=ctx)
    finally:
        if models.detector is None:
            _unload(det)
    tracks = T.link_keyframes(
        kf,
        n_frames=n,
        interval=interval,
        frame_area=W * H,
        iou_new=float(p["iou_new"]),
        link_dist=float(p["link_dist"]),
        min_confirm=int(p["min_confirm"]),
        strong_score=float(p["strong_score"]),
        min_area_frac=float(p["min_area_frac"]),
        max_miss=int(p["max_miss"]),
        max_people=int(p["max_people"]),
    )
    ctx.log(
        f"  키프레임 {len(kf)}장에서 사람 {len(tracks)}명 등록: "
        + ", ".join(f"#{t.tid}@{t.first_frame}" for t in tracks)
    )
    predictor = models.sam2
    try:
        if predictor is None:
            ctx.log(f"  SAM 2.1 {p['sam2_size']} 로딩")
            predictor = load_sam2_video_predictor(models_cfg(), p["sam2_size"])
        return T.track_sam2(
            predictor,
            str(frames_dir),
            tracks,
            n_frames=n,
            height=H,
            width=W,
            interval=interval,
            offload_state=bool(p["sam2_offload_state"]),
            segment=int(p.get("sam2_segment") or 0),
            ctx=ctx,
        )
    finally:
        if models.sam2 is None:
            del predictor
            _unload(None)


def _video_depth(load_frame, load_mask, n, W, H, d3: Path, p: dict, models: PersonModels, ctx) -> dict:
    """프레임별 inverse depth 를 저장하고, 사람 픽셀 기준 정규화 값(모든 프레임 중앙값)을 돌려준다.
    VDA 는 한 번에 넣은 묶음 안에서만 눈금이 일정하므로, 긴 영상은 겹치게 나눠 넣고 겹친 구간으로 눈금을 맞춘다."""
    sw = min(W, int(p["depth_store_width"]))
    sh = max(2, int(round(H * sw / W)))
    per: list[dict] = []
    if not models.use_depth:
        z = np.zeros((sh, sw), np.float16)
        for i in range(n):
            np.save(d3 / f"{i:05d}.npy", z)
        return {"lo": 0.0, "hi": 1.0, "t_front": 1.0, "t_mean": 1.0}
    est = models.depth
    own = est is None
    if own:
        from vr180.models.depth_vda import get_video_depth_estimator

        est = get_video_depth_estimator(p["video_depth_model"], models_cfg())
        if est is None:
            return _video_depth(load_frame, load_mask, n, W, H, d3, p, PersonModels(use_depth=False), ctx)
    group, ov = max(8, int(p["depth_group"])), max(0, int(p["depth_overlap"]))
    prev_tail: dict[int, np.ndarray] = {}
    try:
        with est:
            start = 0
            while start < n:
                s = max(0, start - (ov if start > 0 else 0))
                e = min(n, start + group)
                invs = est.inverse_depth_sequence([load_frame(i) for i in range(s, e)])
                invs = [
                    cv2.resize(x.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) for x in invs
                ]
                if prev_tail:  # 겹친 구간으로 이번 묶음 눈금(a·x+b)을 이전 묶음에 맞춘다
                    xs = np.concatenate([invs[i - s].ravel()[::7] for i in prev_tail])
                    ys = np.concatenate([prev_tail[i].ravel()[::7] for i in prev_tail])
                    A = np.stack([xs, np.ones_like(xs)], 1)
                    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
                    invs = [a * x + b for x in invs]
                prev_tail = {}
                for k, i in enumerate(range(s, e)):
                    if i >= start:
                        np.save(d3 / f"{i:05d}.npy", invs[k].astype(np.float16))
                        m = (
                            cv2.resize(
                                load_mask(i).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST
                            )
                            > 0
                        )
                        st = stereo.depth_stats(invs[k], m)
                        if st:
                            per.append(st)
                    if i >= e - ov:
                        prev_tail[i] = invs[k]
                ctx.progress(e / n, f"깊이 {e}/{n}")
                start = e
    finally:
        if own:
            _unload(est)
    if not per:
        return {"lo": 0.0, "hi": 1.0, "t_front": 1.0, "t_mean": 1.0}
    return {k: float(np.median([x[k] for x in per])) for k in ("lo", "hi", "t_front", "t_mean")}
