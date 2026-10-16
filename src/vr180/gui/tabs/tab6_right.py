"""탭 6 — Right 합성: pano_R, hole 마스크, L/R 토글, 애너글리프. 고급: hole_px, inpaint 모델.
(Stage 6 inpainter 는 큰 구멍 메우기 전용 — Stage 3 outpainter 와 다름.)"""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C
from vr180.models.registry import INPAINTER_NAMES

STAGE = 6


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return [400, "opencv", None, None, None, None]
    is_mono = job.input_mode == "mono"
    pano_l_path = job.path(6, "pano_L.png") if is_mono else job.path(3, "pano_L.png")
    hole_path = job.path(6, "hole_mask_L.png" if is_mono else "hole_mask.png")
    return [
        int(C.get_param(job_id, STAGE, "hole_px", 400)),
        C.get_param(job_id, STAGE, "inpainter", "opencv"),
        C.preview(pano_l_path, 2048),
        C.preview(job.path(6, "pano_R.png"), 2048),
        C.preview_mask(hole_path, 1024),
        C.preview(job.path(6, "anaglyph.png"), 2048),
    ]


def build(shared):
    with gr.Tab("6 Right 합성", id=STAGE):
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            with gr.Row():
                inp = gr.Dropdown(
                    INPAINTER_NAMES,
                    value="opencv",
                    label="큰 구멍 inpainter",
                    info="hole_px 초과 구멍을 메울 모델. opencv/stretch=모델 없음 (기본), lama=빠른 구조 복원, sdxl*/flux=디테일 생성. "
                    "Stage 3 outpainter 와 별개.",
                    elem_classes=["vr180-tip"],
                )
                hole = gr.Slider(
                    0,
                    5000,
                    value=400,
                    step=50,
                    label="hole_px (threshold)",
                    info="이 면적(px²) 이하 구멍은 Telea 로 빠르게, 초과는 inpaint 모델로. 작을수록 모델 호출↑.",
                    elem_classes=["vr180-tip"],
                )
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

        def show(which, l_img, r_img):
            return l_img if which == "L" else r_img

        shared["param_bindings"].extend(
            [
                (hole, STAGE, "hole_px", lambda v: int(v)),
                (inp, STAGE, "inpainter", str),
            ]
        )
        toggle.change(show, [toggle, l_hidden, r_hidden], [lr_view])
        shared["after_load"].append((show, [toggle, l_hidden, r_hidden], [lr_view]))
    return load, outputs
