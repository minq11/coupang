"""탭 0 — 입력: L/R 원본, 수직 시차 히스토그램, 경고. 파라미터 swap_lr, rectify."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C
from vr180.util import imageio

STAGE = 0
RUN_LABEL = "▶ 실행 (Stage 0)"


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
    with gr.Tab("0 입력", id=STAGE):
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            with gr.Row():
                swap = gr.Checkbox(
                    label="swap_lr",
                    value=False,
                    info="입력이 '우=Left, 좌=Right' 로 배열된 경우 켠다. 보통 꺼둠.",
                    elem_classes=["vr180-tip"],
                )
                rect = gr.Checkbox(
                    label="rectify",
                    value=False,
                    info="L/R 특징점 매칭으로 수직 시차를 보정 (y 방향 평행이동). 세로 어긋남이 있을 때만 켠다.",
                    elem_classes=["vr180-tip"],
                )
        with gr.Row(elem_classes=["vr180-actions"]):
            run_btn = gr.Button(RUN_LABEL, variant="primary", elem_classes=["vr180-run"])
        warn = gr.Markdown()
        with gr.Row():
            l_img = gr.Image(label="L 원본", type="numpy", interactive=False)
            r_img = gr.Image(label="R 원본", type="numpy", interactive=False)
        hist = gr.Image(label="수직 시차 히스토그램", type="numpy", interactive=False)

        outputs = [swap, rect, l_img, r_img, hist, warn]

        shared["all_run_buttons"].append(run_btn)

        def run(job_id, swap_v, rect_v):
            yield from C.run_pipeline(
                job_id, STAGE, STAGE, {STAGE: {"swap_lr": bool(swap_v), "rectify": bool(rect_v)}}
            )

        def _wire():
            others = [b for b in shared["all_run_buttons"] if b is not run_btn]
            run_btn.click(
                C.with_button_lock(run, RUN_LABEL, n_other=len(others)),
                [shared["job_id"], swap, rect],
                [
                    run_btn,
                    *others,
                    shared["log"],
                    shared["log_full"],
                    shared["status"],
                    shared["progress"],
                    shared["stage_tabs"],
                ],
            ).then(load, [shared["job_id"]], outputs)

        shared["after_build_wires"].append(_wire)
    return load, outputs
