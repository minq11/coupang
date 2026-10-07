"""탭 1 — FOV: hfov 슬라이더, 자동 추정 토글, 추정값/신뢰도. '다음 단계 자동 미리보기' 는 Stage 1~2 실행."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 1
RUN_LABEL = "▶ 실행 (Stage 1)"
PREV_LABEL = "▶ 실행 + Stage 2 미리보기"


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return [80.0, True, "_Job 없음_", None]
    return [
        float(C.get_param(job_id, STAGE, "hfov_deg", 80.0)),
        bool(C.get_param(job_id, STAGE, "auto_estimate", True)),
        C.json_md(
            job.path(1, "camera.json"),
            ["hfov_deg", "vfov_deg", "focal_px", "source", "confidence", "auto_hfov_deg", "estimator"],
        ),
        C.preview(job.path(2, "preview_sbs.png"), 2048),
    ]


def build(shared):
    with gr.Tab("1 FOV", id=STAGE):
        with gr.Row():
            hfov = gr.Slider(
                40,
                120,
                value=80,
                step=0.5,
                label="hfov_deg (수평 화각)",
                info="원본 3D 영상의 가로 시야각(°). 결과 전체 왜곡을 좌우. 보통 70~90°, 광각 액션캠은 100°+.",
                elem_classes=["vr180-tip"],
            )
            auto = gr.Checkbox(
                label="GeoCalib 자동 추정",
                value=True,
                info="GeoCalib 모델로 FOV 를 추정해서 수동값보다 우선 사용. 신뢰도가 낮으면 수동값으로 복귀.",
                elem_classes=["vr180-tip"],
            )
        info = gr.Markdown()
        with gr.Row(elem_classes=["vr180-actions"]):
            run_btn = gr.Button(RUN_LABEL, variant="primary", elem_classes=["vr180-run"], scale=1)
            prev_btn = gr.Button(PREV_LABEL, elem_classes=["vr180-run"], scale=1)
        prev = gr.Image(label="Stage 2 미리보기 (L|R, 원본 밖 검정)", type="numpy", interactive=False)
        outputs = [hfov, auto, info, prev]

        def run_1(job_id, h, a):
            yield from C.run_pipeline(job_id, STAGE, 1, {STAGE: {"hfov_deg": float(h), "auto_estimate": bool(a)}})

        def run_12(job_id, h, a):
            yield from C.run_pipeline(job_id, STAGE, 2, {STAGE: {"hfov_deg": float(h), "auto_estimate": bool(a)}})

        run_btn.click(
            C.with_button_lock(run_1, RUN_LABEL),
            [shared["job_id"], hfov, auto],
            [
                run_btn,
                shared["log"],
                shared["log_full"],
                shared["status"],
                shared["progress"],
                shared["stage_tabs"],
            ],
        ).then(load, [shared["job_id"]], outputs)
        prev_btn.click(
            C.with_button_lock(run_12, PREV_LABEL),
            [shared["job_id"], hfov, auto],
            [
                prev_btn,
                shared["log"],
                shared["log_full"],
                shared["status"],
                shared["progress"],
                shared["stage_tabs"],
            ],
        ).then(load, [shared["job_id"]], outputs)
    return load, outputs
