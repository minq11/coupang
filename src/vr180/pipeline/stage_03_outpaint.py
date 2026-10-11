"""Stage 3 — 주변부 RGB 생성 (Left 기준). 원근 타일 방식.

equirect 에 diffusion 을 직접 돌리지 않는다. FOV 90° 가상 카메라 타일을 렌더링 → inpaint → 다시 써넣기.
이미 채운 타일은 다음 타일의 조건이 된다 (중앙 → 좌/우 → 상/하 → 모서리).
"""

from __future__ import annotations

import cv2
import numpy as np

from vr180.geometry import reproject
from vr180.util import imageio

INDEX = 3
NAME = "outpaint"
DIRNAME = "03_outpaint"
PARAMS_KEY = "stage_03_outpaint"
USES_MODELS = True

TILE_HINTS = {  # 프롬프트 뒤에 붙는 방향 힌트
    (0, 1): ", ceiling, upper part of the scene",
    (0, -1): ", floor, ground, lower part of the scene",
}


def _src_name(job) -> str:
    return "C_equi.png" if job.input_mode == "mono" else "L_equi.png"


def _dst_name(job) -> str:
    return "pano_C.png" if job.input_mode == "mono" else "pano_L.png"


def inputs(job):
    return [
        job.path(2, _src_name(job)),
        job.path(2, "extent_mask.png"),
        job.path(2, "extent_fade.png"),
    ]


def outputs(job):
    return [job.path(INDEX, _dst_name(job))]


def _composite(pano: np.ndarray, new_rgb: np.ndarray, weight: np.ndarray) -> None:
    """pano(RGBA uint8, in-place) 에 new_rgb 를 weight(0..1) 로 누적 합성. 이미 꽉 찬 픽셀은 건드리지 않는다."""
    a_old = pano[..., 3].astype(np.float32) / 255.0
    w = np.where(a_old >= 0.999, 0.0, weight).astype(np.float32)
    sel = w > 0
    if not sel.any():
        return
    ao = a_old[sel][:, None]
    ww = w[sel][:, None]
    old = pano[..., :3][sel].astype(np.float32)
    new = new_rgb[sel].astype(np.float32)
    denom = ao + ww
    blended = (old * ao + new * ww) / denom
    pano[..., :3][sel] = np.clip(blended, 0, 255).astype(np.uint8)
    pano[..., 3][sel] = (np.clip(ao[:, 0] + ww[:, 0], 0, 1) * 255).astype(np.uint8)


def run(job, params: dict, ctx) -> dict:
    from vr180.models.registry import get_inpainter

    pano = imageio.read_rgba(job.path(2, _src_name(job)))
    size = pano.shape[0]
    # extent: Stage 2 산출물. 범위 밖은 생성에서 뺀다.
    extent_mask = imageio.read_mask(job.path(2, "extent_mask.png")) > 127
    extent_w = imageio.read_mask(job.path(2, "extent_fade.png")).astype(np.float32) / 255.0
    # Stage 2 meta 에서 extent 색상도 쓰고 싶으면 params 로. 기본 검정.
    p2 = job.params.get("stage_02_reproj", {})
    extent_color = p2.get("extent_color", [0, 0, 0]) or [0, 0, 0]
    tile_min_new_fraction = float(p2.get("tile_min_new_fraction", 0.01))

    tile_res = int(params.get("tile_res", 1024))
    tiles = reproject.make_tile_grid(
        params.get("tile_yaws_deg", [-60, 0, 60]),
        params.get("tile_pitches_deg", [-45, 0, 45]),
        float(params.get("tile_fov_deg", 90.0)),
        tile_res,
    )
    only = set(params.get("only_tiles") or [])
    reuse = bool(params.get("reuse_existing_tiles", False))
    feather = int(params.get("feather_px", 64))
    dilate = int(params.get("mask_dilate_px", 8))
    tiles_dir = job.stage_dir(INDEX) / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tiles_executed: list[str] = []
    tiles_skipped: list[str] = []

    inpainter = get_inpainter(params.get("inpainter", "opencv"), job.model_params())
    inpainter.ctx = ctx  # inpaint 중 denoising step 마다 cancel 체크
    info = inpainter.info()
    ctx.log(f"inpainter={info.name} 타일 {len(tiles)}장 res={tile_res}")
    n = len(tiles)
    with inpainter:
        for i, t in enumerate(tiles):
            ctx.progress(i / n, f"타일 {t.name}")
            # 타일이 equirect 에서 덮는 영역 (bool).
            tile_inside, _ = t.feather_weight(size, 0)
            # "새로 채울 픽셀" = 범위 안 ∧ (아직 알파 0) ∧ 타일 안.
            pano_empty = pano[..., 3] < 255
            new_px = (extent_mask & pano_empty & tile_inside).sum()
            denom = max(int(tile_inside.sum()), 1)
            frac = new_px / denom
            if frac < tile_min_new_fraction and not (only and t.name in only):
                ctx.log(f"  {t.name}: 새 픽셀 {frac * 100:.2f}% < {tile_min_new_fraction * 100:.1f}%, 생략")
                tiles_skipped.append(t.name)
                continue
            rgb, alpha = t.render_from_equi(pano)
            mask = (alpha < 0.999).astype(np.uint8) * 255
            if dilate > 0:
                mask = cv2.dilate(mask, np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8))
            # extent 밖은 생성 마스크에서 뺀다 (범위 밖은 fade 로 채우므로).
            if extent_mask is not None:
                extent_tile = t.render_float_from_equi(extent_mask.astype(np.float32)) > 0.5
                mask = (mask > 0).astype(np.uint8) * extent_tile.astype(np.uint8) * 255
            imageio.write_image(tiles_dir / f"{t.name}_in.png", rgb)
            imageio.write_image(tiles_dir / f"{t.name}_mask.png", mask)
            out_path = tiles_dir / f"{t.name}_out.png"
            if mask.max() == 0:
                ctx.log(f"  {t.name}: 빈칸 없음, 건너뜀")
                tiles_skipped.append(t.name)
                result = rgb
            elif reuse and out_path.exists() and (not only or t.name not in only):
                ctx.log(f"  {t.name}: 기존 결과 재사용")
                result = imageio.read_rgb(out_path)
            elif only and t.name not in only and out_path.exists():
                ctx.log(f"  {t.name}: only_tiles 에 없음, 기존 결과 사용")
                result = imageio.read_rgb(out_path)
            else:
                hint = TILE_HINTS.get((0, int(np.sign(t.pitch_deg))), "") if t.yaw_deg == 0 else ""
                # 타일별 override 가 있으면 그걸로 교체 (hint 는 유지)
                tile_overrides = params.get("tile_prompt_overrides") or {}
                base_prompt = tile_overrides.get(t.name) or str(params.get("prompt", ""))
                prompt = base_prompt + hint
                if t.name in tile_overrides:
                    ctx.log(f"  {t.name}: override 프롬프트 사용")
                result = inpainter.inpaint(
                    rgb,
                    mask,
                    prompt=prompt,
                    negative_prompt=str(params.get("negative_prompt", "")),
                    seed=int(params.get("seed", 0)) + i,
                    steps=int(params.get("steps", 30)),
                    guidance=float(params.get("guidance", 7.0)),
                    strength=float(params.get("strength", 1.0)),
                )
                imageio.write_image(out_path, result)
            # 타일 → equirect. 마스크(새로 생성된 곳)만 써넣고, 페더로 경계를 부드럽게.
            new_equi, inside = t.splat_to_equi(result, size)
            _, w = t.feather_weight(size, feather)
            _composite(pano, new_equi, w)
            tiles_executed.append(t.name)
            ctx.check_cancelled()

    # 알파가 0 < a < 255 인 픽셀은 페더 가중치 누적이 덜 된 것일 뿐 색은 이미 있다 → 알파만 채운다.
    # 알파 0 (어떤 타일도 못 덮은 곳) 중 extent_mask 안쪽만 Telea 로 메운다.
    empty_inside = (pano[..., 3] == 0) & extent_mask
    if empty_inside.any():
        frac = float(empty_inside.mean())
        ctx.log(f"타일이 덮지 못한 범위 안 영역 {frac * 100:.2f}% → Telea 보정")
        rgb = pano[..., :3].copy()
        m = empty_inside.astype(np.uint8) * 255
        small = cv2.resize(rgb, (max(8, size // 4), max(8, size // 4)), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(m, (max(8, size // 4), max(8, size // 4)), interpolation=cv2.INTER_NEAREST)
        filled = cv2.resize(
            cv2.inpaint(small, ms, 5, cv2.INPAINT_TELEA), (size, size), interpolation=cv2.INTER_CUBIC
        )
        rgb[empty_inside] = filled[empty_inside]
        pano[..., :3] = rgb
    # extent 페이드 블렌드: rgb = gen × w + color × (1 − w). 바깥 (w == 0) 은 color.
    color = np.asarray(extent_color, dtype=np.float32).reshape(1, 1, 3)
    w3 = extent_w[..., None]
    rgb = pano[..., :3].astype(np.float32)
    pano[..., :3] = np.clip(rgb * w3 + color * (1.0 - w3), 0, 255).astype(np.uint8)
    pano[..., 3] = 255
    imageio.write_image(job.path(INDEX, _dst_name(job)), pano)
    return {
        "model": info.as_dict(),
        "tiles": [t.name for t in tiles],
        "tiles_executed": tiles_executed,
        "tiles_skipped": tiles_skipped,
    }
