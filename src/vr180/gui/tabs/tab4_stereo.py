"""탭 4 — 중앙 disparity: 컬러맵, 원본과 오버레이 슬라이더. 모델 선택, max disparity."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 4
RUN_LABEL = "▶ 실행 (Stage 4)"


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return ["raft_stereo", 256, None, None, "_Job 없음_"]
    lp, dp = job.path(2, "L_equi.png"), job.path(4, "disp_center_preview.png")
    slider = (C.preview(lp, 2048), C.preview(dp, 2048)) if lp.exists() and dp.exists() else None
    return [
        C.get_param(job_id, STAGE, "matcher", "raft_stereo"),
        int(C.get_param(job_id, STAGE, "max_disparity", 256)),
        slider,
        C.preview(job.path(4, "disp_center_px_preview.png"), 1024),
        C.json_md(
            job.path(4, "disp_info.json"),
            ["disp_px_median", "disp_px_p95", "disp_rad_max", "valid_fraction", "focal_px"],
        ),
    ]


def build(shared):
    with gr.Tab("4 중앙 disparity", id=STAGE):
        with gr.Row():
            matcher = gr.Dropdown(
                ["sgbm", "raft_stereo", "raft_flow"],
                value="raft_stereo",
                label="stereo matcher",
                info="sgbm=OpenCV, raft_stereo=RAFT-Stereo (품질↑), raft_flow=optical flow 변형.",
                elem_classes=["vr180-tip"],
            )
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            maxd = gr.Slider(
                32,
                1024,
                value=256,
                step=16,
                label="max disparity (px)",
                info="SGBM 탐색 범위 상한. 가까이 있는 물체가 많으면 늘려야 함. RAFT 는 영향 적음.",
                elem_classes=["vr180-tip"],
            )
        with gr.Row(elem_classes=["vr180-actions"]):
            run_btn = gr.Button(RUN_LABEL, variant="primary", elem_classes=["vr180-run"])
        slider = gr.ImageSlider(label="L_equi ↔ disparity (각도, equirect)", type="numpy", interactive=False)
        with gr.Row():
            px = gr.Image(label="disparity (px, 원근)", type="numpy", interactive=False)
            info = gr.Markdown()
        outputs = [matcher, maxd, slider, px, info]

        def run(job_id, m, d):
            yield from C.run_pipeline(job_id, STAGE, STAGE, {STAGE: {"matcher": m, "max_disparity": int(d)}})

        run_btn.click(
            C.with_button_lock(run, RUN_LABEL),
            [shared["job_id"], matcher, maxd],
            [
                run_btn,
                shared["log"],
                shared["log_full"],
                shared["status"],
                shared["progress"],
                shared["stage_tabs"],
            ],
        ).then(load, [shared["job_id"]], outputs)
    return load, outputs
