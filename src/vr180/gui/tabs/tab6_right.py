"""탭 6 — Right 합성: pano_R, hole 마스크, L/R 토글, 애너글리프. hole_px, inpaint 모델."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 6


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return [400, "opencv", None, None, None, None]
    return [
        int(C.get_param(job_id, STAGE, "hole_px", 400)),
        C.get_param(job_id, STAGE, "inpainter", "opencv"),
        C.preview(job.path(3, "pano_L.png"), 2048),
        C.preview(job.path(6, "pano_R.png"), 2048),
        C.preview_mask(job.path(6, "hole_mask.png"), 1024),
        C.preview(job.path(6, "anaglyph.png"), 2048),
    ]


def build(shared):
    with gr.Tab("6 Right 합성"):
        with gr.Row():
            hole = gr.Slider(0, 5000, value=400, step=50, label="hole_px (이하: Telea, 초과: inpaint 모델)")
            inp = gr.Dropdown(["opencv", "sdxl", "flux"], value="opencv", label="큰 구멍 inpainter")
            run_btn = gr.Button("실행 (Stage 6 부터)", variant="primary")
        toggle = gr.Radio(["L", "R"], value="L", label="L/R 토글 (번갈아 눌러 시차 방향 확인)")
        lr_view = gr.Image(label="pano (L/R)", type="numpy", interactive=False)
        l_hidden = gr.State(None)
        r_hidden = gr.State(None)
        with gr.Row():
            holes = gr.Image(label="hole 마스크", type="numpy", interactive=False)
            ana = gr.Image(
                label="애너글리프 (red=L, cyan=R) — 모니터용 대략 확인", type="numpy", interactive=False
            )
        outputs = [hole, inp, l_hidden, r_hidden, holes, ana]

        def run(job_id, h, i):
            yield from C.run_pipeline(job_id, STAGE, STAGE, {STAGE: {"hole_px": int(h), "inpainter": i}})

        def show(which, l_img, r_img):
            return l_img if which == "L" else r_img

        run_btn.click(run, [shared["job_id"], hole, inp], [shared["log"], shared["status"]]).then(
            load, [shared["job_id"]], outputs
        ).then(show, [toggle, l_hidden, r_hidden], [lr_view])
        toggle.change(show, [toggle, l_hidden, r_hidden], [lr_view])
        shared["after_load"].append((show, [toggle, l_hidden, r_hidden], [lr_view]))
    return load, outputs
