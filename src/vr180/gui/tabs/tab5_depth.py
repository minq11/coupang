"""탭 5 — depth 정합: mono depth, 피팅 산점도, disp_full. band/blend/극지방 클램프/a,b 오버라이드."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 5
RUN_LABEL = "▶ 실행 (Stage 5)"
KEYS = [
    "estimator",
    "band_deg",
    "blend_deg",
    "polar_start_deg",
    "polar_min_disp",
    "override_a",
    "override_b",
    "depth_strength",
    "depth_map_path",
]
DEF = {
    "estimator": "da2_large",
    "band_deg": 5.0,
    "blend_deg": 8.0,
    "polar_start_deg": 75.0,
    "polar_min_disp": 0.0,
    "override_a": None,
    "override_b": None,
    "depth_strength": 0.02,
    "depth_map_path": "",
}


def load(job_id):
    import gradio as gr

    job = C.get_job(job_id)
    vals = [C.get_param(job_id, STAGE, k, DEF[k]) for k in KEYS]
    # None 들을 UI 가 받을 수 있는 형태로 (문자열 비어있음) 변환
    vals[7] = float(vals[7]) if vals[7] is not None else DEF["depth_strength"]
    vals[8] = vals[8] or ""
    if job is None:
        return vals + [None, None, None, "_Job 없음_", gr.update(), gr.update()]
    is_mono = job.input_mode == "mono"
    # mono: sbs 전용 (band/blend/override/polar) 숨김, mono 전용 (depth_strength/map_path) 표시
    sbs_vis = gr.update(visible=not is_mono)
    mono_vis = gr.update(visible=is_mono)
    info_keys = (
        ["depth_strength", "disp_full_max", "model"]
        if is_mono
        else ["a", "b", "inlier_ratio", "fit_source", "num_band_samples", "disp_full_max", "warning"]
    )
    return vals + [
        C.preview(job.path(5, "mono_preview.png"), 1024),
        C.preview(job.path(5, "fit_scatter.png"), 640) if not is_mono else None,
        C.preview(job.path(5, "disp_full_preview.png"), 2048),
        C.json_md(job.path(5, "fit_report.json"), info_keys),
        sbs_vis,
        mono_vis,
    ]


def build(shared):
    with gr.Tab("5 depth 정합", id=STAGE):
        with gr.Row():
            est = gr.Dropdown(
                ["none", "da2_small", "da2_base", "da2_large"],
                value="da2_large",
                label="mono depth",
                info="Depth-Anything v2. large=품질 최고/느림, none=모델 없이 (sbs: stereo 외삽, mono: 평면).",
                elem_classes=["vr180-tip"],
            )
        # mono 전용
        with gr.Group(visible=False) as mono_opts:
            gr.Markdown("**📷 mono 모드** — 피팅 대신 depth_strength 와 백분위 정규화로 바로 disparity 를 만든다.")
            with gr.Row():
                depth_strength = gr.Slider(
                    0.0,
                    0.06,
                    value=0.02,
                    step=0.002,
                    label="depth_strength (각도 disparity 최댓값, rad)",
                    info="≈1.1° (0.02rad) 가 Quest 에서 편한 수준. 크면 입체감↑ 피로↑.",
                    elem_classes=["vr180-tip"],
                )
                depth_map_path = gr.Textbox(
                    value="",
                    label="외부 깊이 지도 경로 (선택)",
                    info="PNG/npy. 비우면 Depth Anything 으로 추정.",
                    elem_classes=["vr180-tip"],
                )
        # sbs 전용 고급 설정
        with gr.Accordion("고급 설정 (sbs 전용, 잘 안 바꿈)", open=False) as sbs_opts:
            with gr.Row():
                band = gr.Slider(
                    1, 20, value=5, step=0.5, label="band_deg (피팅 띠)",
                    info="원본 경계 안쪽에서 stereo↔mono 를 선형 피팅할 띠 폭(°).",
                    elem_classes=["vr180-tip"],
                )
                blend = gr.Slider(
                    1, 30, value=8, step=0.5, label="blend_deg (경계 블렌딩)",
                    info="원본/외삽 경계를 섞을 폭(°).",
                    elem_classes=["vr180-tip"],
                )
            with gr.Row():
                pstart = gr.Slider(
                    45, 90, value=75, step=1, label="극지방 완화 시작 (°)",
                    info="이 위도 이상부터 disparity 를 선형으로 줄임.",
                    elem_classes=["vr180-tip"],
                )
                pmin = gr.Number(
                    value=0.0, label="극에서 최소 disparity (rad)",
                    info="극점에서 남길 각도 disparity. 0 이면 완전 평면.",
                    elem_classes=["vr180-tip"],
                )
                oa = gr.Number(
                    value=None, label="a 오버라이드",
                    info="피팅식 disp = a·mono + b 의 기울기 a 수동 지정.",
                    elem_classes=["vr180-tip"],
                )
                ob = gr.Number(
                    value=None, label="b 오버라이드",
                    info="피팅식의 절편 b 수동 지정.",
                    elem_classes=["vr180-tip"],
                )
        with gr.Row(elem_classes=["vr180-actions"]):
            run_btn = gr.Button(RUN_LABEL, variant="primary", elem_classes=["vr180-run"])
        info = gr.Markdown()
        with gr.Row():
            mono = gr.Image(label="mono inverse depth", type="numpy", interactive=False)
            scatter = gr.Image(label="피팅 산점도 (sbs 전용)", type="numpy", interactive=False)
        full = gr.Image(label="disp_full (각도 disparity, 180°)", type="numpy", interactive=False)
        comps = [est, band, blend, pstart, pmin, oa, ob, depth_strength, depth_map_path]
        outputs = comps + [mono, scatter, full, info, sbs_opts, mono_opts]

        def run(job_id, *v):
            d = dict(zip(KEYS, v, strict=True))
            d["override_a"] = None if d["override_a"] in (None, "") else float(d["override_a"])
            d["override_b"] = None if d["override_b"] in (None, "") else float(d["override_b"])
            d["depth_strength"] = float(d["depth_strength"])
            d["depth_map_path"] = (d["depth_map_path"] or "").strip() or None
            yield from C.run_pipeline(job_id, STAGE, STAGE, {STAGE: d})

        shared["all_run_buttons"].append(run_btn)

        def _wire():
            others = [b for b in shared["all_run_buttons"] if b is not run_btn]
            run_btn.click(
                C.with_button_lock(run, RUN_LABEL, n_other=len(others)),
                [shared["job_id"], *comps],
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
