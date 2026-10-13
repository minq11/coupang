"""탭 3 — 주변부 생성: 타일 그리드(전/후), pano. 고급: negative/seed/steps/guidance/tile_fov/feather/tile_res.
   inpainter · prompt 는 상단 '주요 옵션' 으로 이동. 외부 수정본 업로드는 유지."""

from __future__ import annotations

import shutil

import gradio as gr

from vr180.geometry.reproject import make_tile_grid
from vr180.gui import common as C
from vr180.util import imageio

STAGE = 3
KEYS = ["negative_prompt", "seed", "steps", "guidance", "tile_fov_deg", "feather_px", "tile_res"]
DEF = {
    "negative_prompt": "text, watermark, people",
    "seed": 42,
    "steps": 30,
    "guidance": 30.0,
    "tile_fov_deg": 90.0,
    "feather_px": 64,
    "tile_res": 1024,
}


def tile_names(job_id):
    job = C.get_job(job_id)
    p = job.stage_params(STAGE) if job else {}
    return [
        t.name
        for t in make_tile_grid(
            p.get("tile_yaws_deg", [-60, 0, 60]), p.get("tile_pitches_deg", [-45, 0, 45]), 90, 64
        )
    ]


def load(job_id):
    job = C.get_job(job_id)
    vals = [C.get_param(job_id, STAGE, k, DEF[k]) for k in KEYS]
    if job is None:
        return vals + [[], None]
    td = job.stage_dir(3) / "tiles"
    gallery = []
    for n in tile_names(job_id):
        for suf in ("in", "out"):
            p = td / f"{n}_{suf}.png"
            if p.exists():
                gallery.append((imageio.downscale_for_preview(imageio.read_rgb(p), 512), f"{n} {suf}"))
    pano_name = "pano_C.png" if job.input_mode == "mono" else "pano_L.png"
    return vals + [gallery, C.preview(job.path(3, pano_name), 2048)]


def build(shared):
    with gr.Tab("3 주변부 생성", id=STAGE):
        gr.Markdown("inpainter · 프롬프트는 **상단 '주요 옵션'** 에서 설정. 아래는 네거티브·seed 등 고급 조절.")
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            neg = gr.Textbox(
                label="네거티브",
                value=DEF["negative_prompt"],
                info="피하고 싶은 요소. 사람/워터마크/글자 등을 명시하면 결과가 깔끔해진다.",
                elem_classes=["vr180-tip"],
            )
            with gr.Row():
                seed = gr.Number(
                    value=42,
                    label="seed",
                    precision=0,
                    info="같은 seed → 같은 결과 (재현용). 결과가 마음에 안 들면 숫자를 바꿔본다.",
                    elem_classes=["vr180-tip"],
                )
                steps = gr.Slider(
                    1, 60, value=30, step=1,
                    label="steps",
                    info="디노이즈 단계 수. 많을수록 품질↑ 속도↓. SDXL 30, FLUX 20~30 권장.",
                    elem_classes=["vr180-tip"],
                )
                guidance = gr.Slider(
                    0, 40, value=30.0, step=0.5,
                    label="guidance",
                    info="프롬프트 반영 강도. FLUX 는 30 근처, SDXL 는 7 근처 권장.",
                    elem_classes=["vr180-tip"],
                )
            with gr.Row():
                tile_fov = gr.Slider(
                    60, 120, value=90, step=1,
                    label="타일 FOV (°)",
                    info="각 타일이 포괄하는 가로 시야각. 커질수록 왜곡은 늘고 이음새는 줄어든다.",
                    elem_classes=["vr180-tip"],
                )
                feather = gr.Slider(
                    0, 256, value=64, step=4,
                    label="페더 px",
                    info="타일 가장자리 블렌딩 폭. 크면 부드럽지만 디테일이 뭉개질 수 있음.",
                    elem_classes=["vr180-tip"],
                )
                tile_res = gr.Dropdown(
                    [512, 768, 1024], value=1024,
                    label="타일 해상도",
                    info="디퓨전 모델이 받는 입력 해상도. 1024 가 품질/VRAM 균형.",
                    elem_classes=["vr180-tip"],
                )
        with gr.Accordion("외부에서 수정한 pano_L.png 불러오기", open=False):
            upload = gr.File(label="pano_L.png (덮어쓰기 → Stage 5 부터 재실행)")
            with gr.Row(elem_classes=["vr180-actions"]):
                upload_btn = gr.Button("📥 불러오기")
        gallery = gr.Gallery(label="타일 (in / out)", columns=4, height=360)
        pano = gr.Image(label="pano (sbs: pano_L, mono: pano_C)", type="numpy", interactive=False)
        comps = [neg, seed, steps, guidance, tile_fov, feather, tile_res]
        outputs = comps + [gallery, pano]

        def do_upload(job_id, f):
            job = C.get_job(job_id)
            if job is None or f is None:
                raise gr.Error("Job 과 파일이 필요합니다")
            name = "pano_C.png" if job.input_mode == "mono" else "pano_L.png"
            dst = job.path(3, name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = imageio.read_rgba(f)
            if img.shape[0] != img.shape[1]:
                raise gr.Error(f"{name} 은 정사각이어야 합니다")
            shutil.copy2(f, dst)
            return f"덮어씀: {dst} — Stage 5 부터 재실행하면 반영됩니다", C.status_md(job_id)

        shared["param_bindings"].extend([
            (neg, STAGE, "negative_prompt", str),
            (seed, STAGE, "seed", lambda v: int(v) if v is not None else 42),
            (steps, STAGE, "steps", lambda v: int(v)),
            (guidance, STAGE, "guidance", float),
            (tile_fov, STAGE, "tile_fov_deg", float),
            (feather, STAGE, "feather_px", lambda v: int(v)),
            (tile_res, STAGE, "tile_res", lambda v: int(v)),
        ])
        upload_btn.click(do_upload, [shared["job_id"], upload], [shared["log"], shared["status"]]).then(
            load, [shared["job_id"]], outputs
        )
    return load, outputs
