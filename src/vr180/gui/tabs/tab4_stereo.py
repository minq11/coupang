"""탭 4 — 중앙 disparity: 컬러맵, 원본과 오버레이. 고급: matcher, max disparity."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 4


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return ["raft_stereo", 256, None, None, "_Job 없음_"]
    if job.input_mode == "mono":
        return [
            "raft_stereo",
            256,
            None,
            None,
            "➖ **mono 모드에서는 Stage 4를 건너뜁니다.** "
            "스테레오 매칭 대신 Stage 5의 mono depth 가 직접 쓰입니다.",
        ]
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
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            with gr.Row():
                matcher = gr.Dropdown(
                    ["sgbm", "raft_stereo", "raft_flow"],
                    value="raft_stereo",
                    label="stereo matcher",
                    info="sgbm=OpenCV, raft_stereo=RAFT-Stereo (품질↑), raft_flow=optical flow 변형.",
                    elem_classes=["vr180-tip"],
                )
                maxd = gr.Slider(
                    32, 1024, value=256, step=16,
                    label="max disparity (px)",
                    info="SGBM 탐색 범위 상한. 가까이 있는 물체가 많으면 늘려야 함. RAFT 는 영향 적음.",
                    elem_classes=["vr180-tip"],
                )
        slider = gr.ImageSlider(label="L_equi ↔ disparity (각도, equirect)", type="numpy", interactive=False)
        with gr.Row():
            px = gr.Image(label="disparity (px, 원근)", type="numpy", interactive=False)
            info = gr.Markdown()
        outputs = [matcher, maxd, slider, px, info]

        shared["param_bindings"].extend([
            (matcher, STAGE, "matcher", str),
            (maxd, STAGE, "max_disparity", lambda v: int(v)),
        ])
    return load, outputs
