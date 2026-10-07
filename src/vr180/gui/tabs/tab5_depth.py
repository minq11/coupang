"""탭 5 — depth 정합: mono depth, 피팅 산점도, disp_full. band/blend/극지방 클램프/a,b 오버라이드."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 5
KEYS = ["estimator", "band_deg", "blend_deg", "polar_start_deg", "polar_min_disp", "override_a", "override_b"]
DEF = {
    "estimator": "da2_large",
    "band_deg": 5.0,
    "blend_deg": 8.0,
    "polar_start_deg": 75.0,
    "polar_min_disp": 0.0,
    "override_a": None,
    "override_b": None,
}


def load(job_id):
    job = C.get_job(job_id)
    vals = [C.get_param(job_id, STAGE, k, DEF[k]) for k in KEYS]
    if job is None:
        return vals + [None, None, None, "_Job 없음_"]
    return vals + [
        C.preview(job.path(5, "mono_preview.png"), 1024),
        C.preview(job.path(5, "fit_scatter.png"), 640),
        C.preview(job.path(5, "disp_full_preview.png"), 2048),
        C.json_md(
            job.path(5, "fit_report.json"),
            ["a", "b", "inlier_ratio", "fit_source", "num_band_samples", "disp_full_max", "warning"],
        ),
    ]


def build(shared):
    with gr.Tab("5 depth 정합"):
        with gr.Row():
            est = gr.Dropdown(
                ["none", "da2_small", "da2_base", "da2_large"],
                value="da2_large",
                label="mono depth (none = stereo 외삽, 모델 없음)",
            )
            band = gr.Slider(1, 20, value=5, step=0.5, label="band_deg (피팅 띠)")
            blend = gr.Slider(1, 30, value=8, step=0.5, label="blend_deg (경계 블렌딩)")
        with gr.Row():
            pstart = gr.Slider(45, 90, value=75, step=1, label="극지방 완화 시작 (°)")
            pmin = gr.Number(value=0.0, label="극에서 최소 disparity (rad)")
            oa = gr.Number(value=None, label="a 오버라이드 (비우면 피팅)")
            ob = gr.Number(value=None, label="b 오버라이드")
            run_btn = gr.Button("실행 (Stage 5 부터)", variant="primary")
        info = gr.Markdown()
        with gr.Row():
            mono = gr.Image(label="mono inverse depth", type="numpy", interactive=False)
            scatter = gr.Image(label="피팅 산점도 (x: mono, y: stereo)", type="numpy", interactive=False)
        full = gr.Image(label="disp_full (각도 disparity, 180°)", type="numpy", interactive=False)
        comps = [est, band, blend, pstart, pmin, oa, ob]
        outputs = comps + [mono, scatter, full, info]

        def run(job_id, *v):
            d = dict(zip(KEYS, v, strict=True))
            d["override_a"] = None if d["override_a"] in (None, "") else float(d["override_a"])
            d["override_b"] = None if d["override_b"] in (None, "") else float(d["override_b"])
            yield from C.run_pipeline(job_id, STAGE, STAGE, {STAGE: d})

        run_btn.click(run, [shared["job_id"], *comps], [shared["log"], shared["status"]]).then(
            load, [shared["job_id"]], outputs
        )
    return load, outputs
