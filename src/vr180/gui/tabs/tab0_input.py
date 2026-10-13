"""탭 0 — 입력: L/R 원본, 수직 시차 히스토그램, 경고. 고급(sbs 전용): swap_lr, rectify."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C
from vr180.util import imageio

STAGE = 0


def load(job_id):
    import gradio as gr

    job = C.get_job(job_id)
    if job is None:
        return [False, False, None, None, None, "_Job 없음_", gr.update()]
    mode = job.input_mode
    rep_p = job.path(0, "rectify_report.json")
    hist, warn = None, "_아직 실행 안 함_"
    if mode == "mono":
        if rep_p.exists():
            warn = "📷 **mono 모드** — 자르기·수직 보정 없음, 가운데 뷰 1장을 그대로 사용합니다."
        c = C.preview(job.path(0, "C.png"), 1024)
        return [
            False,
            False,
            c,
            None,
            None,
            warn,
            gr.update(visible=False),
        ]
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
        gr.update(visible=True),
    ]


def build(shared):
    with gr.Tab("0 입력", id=STAGE):
        with gr.Accordion("고급 설정 (sbs 전용, 잘 안 바꿈)", open=False) as sbs_opts:
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
        warn = gr.Markdown()
        with gr.Row():
            l_img = gr.Image(label="원본 (sbs: L, mono: 전체)", type="numpy", interactive=False)
            r_img = gr.Image(label="R 원본 (mono 에서는 비어 있음)", type="numpy", interactive=False)
        hist = gr.Image(label="수직 시차 히스토그램 (sbs 전용)", type="numpy", interactive=False)

        outputs = [swap, rect, l_img, r_img, hist, warn, sbs_opts]

        shared["param_bindings"].extend([
            (swap, STAGE, "swap_lr", bool),
            (rect, STAGE, "rectify", bool),
        ])
    return load, outputs
