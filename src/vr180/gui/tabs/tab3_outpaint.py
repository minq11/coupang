"""탭 3 — 주변부 생성: 타일 그리드(전/후), pano_L. 프롬프트/seed/steps/guidance/타일 FOV/페더, 타일 개별 재생성, 외부 수정본."""

from __future__ import annotations

import shutil

import gradio as gr

from vr180.geometry.reproject import make_tile_grid
from vr180.gui import common as C
from vr180.util import imageio

STAGE = 3
KEYS = [
    "inpainter",
    "prompt",
    "negative_prompt",
    "seed",
    "steps",
    "guidance",
    "tile_fov_deg",
    "feather_px",
    "tile_res",
]
DEF = {
    "inpainter": "opencv",
    "prompt": "",
    "negative_prompt": "text, watermark, people",
    "seed": 42,
    "steps": 30,
    "guidance": 7.0,
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
        return vals + [gr.update(choices=[], value=[]), [], None]
    td = job.stage_dir(3) / "tiles"
    gallery = []
    for n in tile_names(job_id):
        for suf in ("in", "out"):
            p = td / f"{n}_{suf}.png"
            if p.exists():
                gallery.append((imageio.downscale_for_preview(imageio.read_rgb(p), 512), f"{n} {suf}"))
    return vals + [
        gr.update(choices=tile_names(job_id), value=[]),
        gallery,
        C.preview(job.path(3, "pano_L.png"), 2048),
    ]


def build(shared):
    with gr.Tab("3 주변부 생성"):
        with gr.Row():
            inpainter = gr.Dropdown(
                ["opencv", "sdxl", "flux"], value="opencv", label="inpainter (opencv = 모델 없음)"
            )
            tile_res = gr.Dropdown([512, 768, 1024], value=1024, label="타일 해상도")
            tile_fov = gr.Slider(60, 120, value=90, step=1, label="타일 FOV (°)")
            feather = gr.Slider(0, 256, value=64, step=4, label="페더 px")
        prompt = gr.Textbox(label="프롬프트 (장면 묘사: 실내/실외, 조명, 바닥·천장 재질)", lines=2)
        neg = gr.Textbox(label="네거티브", value=DEF["negative_prompt"])
        with gr.Row():
            seed = gr.Number(value=42, label="seed", precision=0)
            steps = gr.Slider(1, 60, value=30, step=1, label="steps")
            guidance = gr.Slider(0, 40, value=7.0, step=0.5, label="guidance (FLUX 는 30 권장)")
        with gr.Row():
            run_btn = gr.Button("전체 실행 (Stage 3 부터)", variant="primary")
            only = gr.CheckboxGroup(choices=[], label="선택 타일만 재생성")
            only_btn = gr.Button("선택 타일만 재생성")
        with gr.Row():
            upload = gr.File(label="외부 수정본 pano_L.png 불러오기 (덮어쓰기 → Stage 5 부터 재실행)")
            upload_btn = gr.Button("불러오기")
        gallery = gr.Gallery(label="타일 (in / out)", columns=4, height=360)
        pano = gr.Image(label="pano_L", type="numpy", interactive=False)
        comps = [inpainter, prompt, neg, seed, steps, guidance, tile_fov, feather, tile_res]
        outputs = comps + [only, gallery, pano]

        def _vals(*v):
            d = dict(zip(KEYS, v, strict=True))
            d["seed"] = int(d["seed"])
            d["steps"] = int(d["steps"])
            d["feather_px"] = int(d["feather_px"])
            d["tile_res"] = int(d["tile_res"])
            return d

        def run_all(job_id, *v):
            d = _vals(*v)
            d["only_tiles"] = []
            d["reuse_existing_tiles"] = False
            yield from C.run_pipeline(job_id, STAGE, None, {STAGE: d})

        def run_only(job_id, sel, *v):
            if not sel:
                raise gr.Error("재생성할 타일을 고르세요")
            d = _vals(*v)
            d["only_tiles"] = list(sel)
            d["reuse_existing_tiles"] = True
            yield from C.run_pipeline(job_id, STAGE, STAGE, {STAGE: d})

        def do_upload(job_id, f):
            job = C.get_job(job_id)
            if job is None or f is None:
                raise gr.Error("Job 과 파일이 필요합니다")
            dst = job.path(3, "pano_L.png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = imageio.read_rgba(f)
            if img.shape[0] != img.shape[1]:
                raise gr.Error("pano_L 은 정사각이어야 합니다")
            shutil.copy2(f, dst)
            return f"덮어씀: {dst} — Stage 5 부터 재실행하면 반영됩니다", C.status_md(job_id)

        run_btn.click(run_all, [shared["job_id"], *comps], [shared["log"], shared["status"]]).then(
            load, [shared["job_id"]], outputs
        )
        only_btn.click(run_only, [shared["job_id"], only, *comps], [shared["log"], shared["status"]]).then(
            load, [shared["job_id"]], outputs
        )
        upload_btn.click(do_upload, [shared["job_id"], upload], [shared["log"], shared["status"]]).then(
            load, [shared["job_id"]], outputs
        )
    return load, outputs
