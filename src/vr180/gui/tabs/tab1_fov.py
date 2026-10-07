"""탭 1 — FOV: hfov 슬라이더, 자동 추정 토글, 추정값/신뢰도. '다음 단계 자동 미리보기' 는 Stage 1~2 실행."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 1


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return [80.0, False, "_Job 없음_", None]
    return [
        float(C.get_param(job_id, STAGE, "hfov_deg", 80.0)),
        bool(C.get_param(job_id, STAGE, "auto_estimate", False)),
        C.json_md(
            job.path(1, "camera.json"),
            ["hfov_deg", "vfov_deg", "focal_px", "source", "confidence", "auto_hfov_deg", "estimator"],
        ),
        C.preview(job.path(2, "preview_sbs.png"), 2048),
    ]


def build(shared):
    with gr.Tab("1 FOV"):
        with gr.Row():
            hfov = gr.Slider(40, 120, value=80, step=0.5, label="hfov_deg (수평 화각)")
            auto = gr.Checkbox(label="GeoCalib 자동 추정 (설치 필요)", value=False)
        with gr.Row():
            run_btn = gr.Button("실행 (Stage 1)", variant="primary")
            prev_btn = gr.Button("실행 + 다음 단계 자동 미리보기 (Stage 1~2)")
        info = gr.Markdown()
        prev = gr.Image(label="Stage 2 미리보기 (L|R, 원본 밖 검정)", type="numpy", interactive=False)
        outputs = [hfov, auto, info, prev]

        def run(job_id, h, a, to):
            yield from C.run_pipeline(
                job_id, STAGE, to, {STAGE: {"hfov_deg": float(h), "auto_estimate": bool(a)}}
            )

        run_btn.click(
            lambda j, h, a: (yield from run(j, h, a, 1)),
            [shared["job_id"], hfov, auto],
            [shared["log"], shared["status"]],
        ).then(load, [shared["job_id"]], outputs)
        prev_btn.click(
            lambda j, h, a: (yield from run(j, h, a, 2)),
            [shared["job_id"], hfov, auto],
            [shared["log"], shared["status"]],
        ).then(load, [shared["job_id"]], outputs)
    return load, outputs
