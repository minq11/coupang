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


def inputs(job):
    return [job.path(2, "L_equi.png")]


def outputs(job):
    return [job.path(INDEX, "pano_L.png")]


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

    pano = imageio.read_rgba(job.path(2, "L_equi.png"))
    size = pano.shape[0]
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

    inpainter = get_inpainter(params.get("inpainter", "opencv"), job.model_params())
    info = inpainter.info()
    ctx.log(f"inpainter={info.name} 타일 {len(tiles)}장 res={tile_res}")
    n = len(tiles)
    with inpainter:
        for i, t in enumerate(tiles):
            ctx.progress(i / n, f"타일 {t.name}")
            rgb, alpha = t.render_from_equi(pano)
            mask = (alpha < 0.999).astype(np.uint8) * 255
            if dilate > 0:
                mask = cv2.dilate(mask, np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8))
            imageio.write_image(tiles_dir / f"{t.name}_in.png", rgb)
            imageio.write_image(tiles_dir / f"{t.name}_mask.png", mask)
            out_path = tiles_dir / f"{t.name}_out.png"
            if mask.max() == 0:
                ctx.log(f"  {t.name}: 빈칸 없음, 건너뜀")
                result = rgb
            elif reuse and out_path.exists() and (not only or t.name not in only):
                ctx.log(f"  {t.name}: 기존 결과 재사용")
                result = imageio.read_rgb(out_path)
            elif only and t.name not in only and out_path.exists():
                ctx.log(f"  {t.name}: only_tiles 에 없음, 기존 결과 사용")
                result = imageio.read_rgb(out_path)
            else:
                hint = TILE_HINTS.get((0, int(np.sign(t.pitch_deg))), "") if t.yaw_deg == 0 else ""
                prompt = str(params.get("prompt", "")) + hint
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
            ctx.check_cancelled()

    # 알파가 0 < a < 255 인 픽셀은 페더 가중치 누적이 덜 된 것일 뿐 색은 이미 있다 → 알파만 채운다.
    # 알파 0 (어떤 타일도 못 덮은 곳, 극 근처 등) 만 Telea 로 메운다.
    empty = pano[..., 3] == 0
    if empty.any():
        frac = float(empty.mean())
        ctx.log(f"타일이 덮지 못한 영역 {frac * 100:.2f}% → Telea 보정")
        rgb = pano[..., :3].copy()
        m = empty.astype(np.uint8) * 255
        small = cv2.resize(rgb, (max(8, size // 4), max(8, size // 4)), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(m, (max(8, size // 4), max(8, size // 4)), interpolation=cv2.INTER_NEAREST)
        filled = cv2.resize(
            cv2.inpaint(small, ms, 5, cv2.INPAINT_TELEA), (size, size), interpolation=cv2.INTER_CUBIC
        )
        rgb[empty] = filled[empty]
        pano[..., :3] = rgb
    pano[..., 3] = 255
    imageio.write_image(job.path(INDEX, "pano_L.png"), pano)
    return {"model": info.as_dict(), "tiles": [t.name for t in tiles]}
