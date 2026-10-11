"""탭 6 — Right 합성: pano_R, hole 마스크, L/R 토글, 애너글리프. hole_px, inpaint 모델."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 6
RUN_LABEL = "▶ 실행 (Stage 6)"


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
        with gr.Row():
            inp = gr.Dropdown(
                ["opencv", "sdxl", "flux"],
                value="opencv",
                label="큰 구멍 inpainter",
                info="hole_px 초과 구멍을 메울 모델. opencv=빠른 Telea (기본), sdxl/flux=디테일 생성.",
                elem_classes=["vr180-tip"],
            )
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            hole = gr.Slider(
                0,
                5000,
                value=400,
                step=50,
                label="hole_px (threshold)",
                info="이 면적(px²) 이하 구멍은 Telea 로 빠르게, 초과는 inpaint 모델로. 작을수록 모델 호출↑.",
                elem_classes=["vr180-tip"],
            )
        with gr.Row(elem_classes=["vr180-actions"]):
            run_btn = gr.Button(RUN_LABEL, variant="primary", elem_classes=["vr180-run"])
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

        shared["all_run_buttons"].append(run_btn)

        def _wire():
            others = [b for b in shared["all_run_buttons"] if b is not run_btn]
            run_btn.click(
                C.with_button_lock(run, RUN_LABEL, n_other=len(others)),
                [shared["job_id"], hole, inp],
                [
                    run_btn,
                    *others,
                    shared["log"],
                    shared["log_full"],
                    shared["status"],
                    shared["progress"],
                    shared["stage_tabs"],
                ],
            ).then(load, [shared["job_id"]], outputs).then(
                show, [toggle, l_hidden, r_hidden], [lr_view]
            )

        shared["after_build_wires"].append(_wire)
        toggle.change(show, [toggle, l_hidden, r_hidden], [lr_view])
        shared["after_load"].append((show, [toggle, l_hidden, r_hidden], [lr_view]))
    return load, outputs
