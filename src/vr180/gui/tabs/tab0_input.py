"""탭 0 — 입력: L/R 원본, 수직 시차 히스토그램, 경고. 파라미터 swap_lr, rectify."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C
from vr180.util import imageio

STAGE = 0


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return [False, False, None, None, None, "_Job 없음_"]
    rep_p = job.path(0, "rectify_report.json")
    hist, warn = None, "_아직 실행 안 함_"
    if rep_p.exists():
        rep = imageio.read_json(rep_p)
        hist = C.histogram_image(
            rep.get("dy_samples", []),
            f"dy = y_R - y_L (px), median {rep.get('dy_median_px', 0):.2f}, n={rep.get('num_matches', 0)}",
        )
        warn = (
            ("⚠️ " + rep["warning"])
            if rep.get("warning")
            else f"✅ 수직 시차 중앙값 {rep.get('dy_median_px', 0):.2f}px (매칭 {rep.get('num_matches', 0)}개, {rep.get('matcher', '')})"
        )
        if rep.get("rectified"):
            warn += f" · rectify 적용: R 을 {rep.get('applied_shift_px', 0):+.2f}px 이동"
    return [
        bool(C.get_param(job_id, STAGE, "swap_lr", False)),
        bool(C.get_param(job_id, STAGE, "rectify", False)),
        C.preview(job.path(0, "L.png"), 1024),
        C.preview(job.path(0, "R.png"), 1024),
        hist,
        warn,
    ]


def build(shared):
    with gr.Tab("0 입력"):
        with gr.Row():
            swap = gr.Checkbox(label="swap_lr (입력이 우=Left)", value=False)
            rect = gr.Checkbox(label="rectify (수직 시차 보정)", value=False)
            run_btn = gr.Button("실행 (Stage 0 부터)", variant="primary")
        warn = gr.Markdown()
        with gr.Row():
            l_img = gr.Image(label="L 원본", type="numpy", interactive=False)
            r_img = gr.Image(label="R 원본", type="numpy", interactive=False)
        hist = gr.Image(label="수직 시차 히스토그램", type="numpy", interactive=False)

        outputs = [swap, rect, l_img, r_img, hist, warn]

        def run(job_id, swap_v, rect_v):
            yield from C.run_pipeline(
                job_id, STAGE, STAGE, {STAGE: {"swap_lr": bool(swap_v), "rectify": bool(rect_v)}}
            )

        run_btn.click(run, [shared["job_id"], swap, rect], [shared["log"], shared["status"]]).then(
            load, [shared["job_id"]], outputs
        )
    return load, outputs
