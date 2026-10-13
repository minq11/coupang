"""탭 2 — 재투영: L/R equirect (원본 밖은 체크무늬), valid mask. 고급: margin_h/v, shape, interpolation.
   size, extent_h, extent_v, fade 는 상단 '주요 옵션' 으로 이동."""

from __future__ import annotations

import shutil

import gradio as gr

from vr180.gui import common as C

STAGE = 2


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return [20.0, 25.0, "ellipse", "lanczos", None, None, None, "_Job 없음_"]
    if job.input_mode == "mono":
        cp = job.path(2, "C_equi.png")
        slider = (
            (C.preview(cp, 2048, checker=True), C.preview(cp, 2048, checker=True))
            if cp.exists()
            else None
        )
    else:
        lp, rp = job.path(2, "L_equi.png"), job.path(2, "R_equi.png")
        slider = (
            (C.preview(lp, 2048, checker=True), C.preview(rp, 2048, checker=True))
            if lp.exists() and rp.exists()
            else None
        )
    md = C.json_md(
        job.path(2, "meta.json"),
        ["extent_h_deg", "extent_v_deg", "extent_fade_deg", "extent_inside_fraction"],
    )
    return [
        float(C.get_param(job_id, STAGE, "extent_margin_h_deg", 20.0)),
        float(C.get_param(job_id, STAGE, "extent_margin_v_deg", 25.0)),
        C.get_param(job_id, STAGE, "extent_shape", "ellipse"),
        C.get_param(job_id, STAGE, "interpolation", "lanczos"),
        slider,
        C.preview_mask(job.path(2, "valid_mask.png"), 1024),
        str(job.path(2, "preview_sbs.png")) if job.path(2, "preview_sbs.png").exists() else None,
        md,
    ]


def build(shared):
    with gr.Tab("2 재투영", id=STAGE):
        gr.Markdown(
            "해상도 S · 생성 범위(extent_h/v) · 페이드는 **상단 '주요 옵션'** 에서 설정하세요. "
            "아래는 거의 안 바꾸는 고급 설정."
        )
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            with gr.Row():
                mh = gr.Slider(
                    0, 50, value=20, step=1, label="수평 여유 margin_h (°, 한쪽)",
                    info="extent_h_deg 가 '전체 수평 생성 범위' 로 상단에서 지정되므로 margin 은 사용되지 않음. "
                    "레거시 호환용. (하위 공식 참고: h = hfov + 2·margin_h)",
                    elem_classes=["vr180-tip"],
                )
                mv = gr.Slider(
                    0, 50, value=25, step=1, label="수직 여유 margin_v (°, 한쪽)",
                    info="extent_v_deg 가 상단에서 지정되므로 margin 은 사용되지 않음. 레거시 호환용.",
                    elem_classes=["vr180-tip"],
                )
                shape = gr.Dropdown(
                    ["ellipse", "rounded_rect"], value="ellipse", label="범위 모양",
                    info="ellipse 가 비네팅처럼 자연스러움.",
                    elem_classes=["vr180-tip"],
                )
            interp = gr.Dropdown(
                ["lanczos", "cubic", "linear", "nearest"],
                value="lanczos",
                label="보간",
                info="재투영 샘플링 커널. lanczos = 품질 최고, nearest = 가장 빠름.",
                elem_classes=["vr180-tip"],
            )
        with gr.Row(elem_classes=["vr180-actions"]):
            export_btn = gr.Button("📤 Quest 미리보기 내보내기")
        extent_info = gr.Markdown()
        slider = gr.ImageSlider(label="L_equi ↔ R_equi — 흰 선 = 생성 범위 경계", type="numpy", interactive=False)
        with gr.Row():
            mask = gr.Image(label="valid_mask", type="numpy", interactive=False)
            prev_file = gr.File(label="preview_sbs.png (다운로드)")
        outputs = [mh, mv, shape, interp, slider, mask, prev_file, extent_info]

        def export(job_id):
            job = C.get_job(job_id)
            if job is None or not job.path(2, "preview_sbs.png").exists():
                raise gr.Error("Stage 2 결과가 없습니다")
            dst = job.stage_dir(7) / "preview_stage2_SBS.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(job.path(2, "preview_sbs.png"), dst)
            return f"내보냄: {dst}"

        shared["param_bindings"].extend([
            (mh, STAGE, "extent_margin_h_deg", float),
            (mv, STAGE, "extent_margin_v_deg", float),
            (shape, STAGE, "extent_shape", str),
            (interp, STAGE, "interpolation", str),
        ])
        export_btn.click(export, [shared["job_id"]], [shared["log"]])
    return load, outputs
