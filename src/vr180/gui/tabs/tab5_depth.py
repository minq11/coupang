"""탭 5 — depth 정합: mono depth, 피팅 산점도, disp_full. 고급(sbs): band/blend/극지방/a,b 오버라이드.
   mono depth 모델 estimator · depth_strength 는 상단 '주요 옵션' 으로 이동."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 5
KEYS = ["band_deg", "blend_deg", "polar_start_deg", "polar_min_disp", "override_a", "override_b", "depth_map_path"]
DEF = {
    "band_deg": 5.0,
    "blend_deg": 8.0,
    "polar_start_deg": 75.0,
    "polar_min_disp": 0.0,
    "override_a": None,
    "override_b": None,
    "depth_map_path": "",
}


def load(job_id):
    import gradio as gr

    job = C.get_job(job_id)
    vals = [C.get_param(job_id, STAGE, k, DEF[k]) for k in KEYS]
    vals[6] = vals[6] or ""  # depth_map_path: None → ""
    if job is None:
        return vals + [None, None, None, "_Job 없음_", gr.update(), gr.update()]
    is_mono = job.input_mode == "mono"
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
        gr.Markdown("mono depth 모델 · depth_strength 는 **상단 '주요 옵션'** 에서 설정.")
        # mono 전용
        with gr.Group(visible=False) as mono_opts:
            gr.Markdown("**📷 mono 모드** — 외부 깊이 지도를 쓰려면 경로 지정. (없으면 Depth Anything 사용)")
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
        info = gr.Markdown()
        with gr.Row():
            mono = gr.Image(label="mono inverse depth", type="numpy", interactive=False)
            scatter = gr.Image(label="피팅 산점도 (sbs 전용)", type="numpy", interactive=False)
        full = gr.Image(label="disp_full (각도 disparity, 180°)", type="numpy", interactive=False)
        # outputs 순서는 load() 반환 순서와 일치해야 함: KEYS 순 + 미리보기들 + 가시성 토글
        outputs = [band, blend, pstart, pmin, oa, ob, depth_map_path, mono, scatter, full, info, sbs_opts, mono_opts]

        shared["param_bindings"].extend([
            (band, STAGE, "band_deg", float),
            (blend, STAGE, "blend_deg", float),
            (pstart, STAGE, "polar_start_deg", float),
            (pmin, STAGE, "polar_min_disp", lambda v: float(v) if v not in (None, "") else 0.0),
            (oa, STAGE, "override_a", lambda v: None if v in (None, "") else float(v)),
            (ob, STAGE, "override_b", lambda v: None if v in (None, "") else float(v)),
            (depth_map_path, STAGE, "depth_map_path", lambda v: (v or "").strip() or None),
        ])
    return load, outputs
