"""탭 1 — FOV: Stage 1 결과 요약 + Stage 2 미리보기. 조절 위젯(hfov/auto)은 상단 '주요 옵션' 으로 이동."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 1


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return ["_Job 없음_", None]
    return [
        C.json_md(
            job.path(1, "camera.json"),
            ["hfov_deg", "vfov_deg", "focal_px", "source", "confidence", "auto_hfov_deg", "estimator"],
        ),
        C.preview(job.path(2, "preview_sbs.png"), 2048),
    ]


def build(shared):
    with gr.Tab("1 FOV", id=STAGE):
        gr.Markdown("hfov · GeoCalib 자동 추정은 **상단 '주요 옵션'** 에서 설정하세요.")
        info = gr.Markdown()
        prev = gr.Image(label="Stage 2 미리보기 (L|R, 원본 밖 검정)", type="numpy", interactive=False)
        outputs = [info, prev]
    return load, outputs
