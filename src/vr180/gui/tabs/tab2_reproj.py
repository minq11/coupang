"""탭 2 — 재투영: L/R equirect (원본 밖은 체크무늬), valid mask. 출력 해상도 S, 보간. Quest 미리보기 내보내기."""

from __future__ import annotations

import shutil

import gradio as gr

from vr180.gui import common as C

STAGE = 2
RUN_LABEL = "▶ 실행 (Stage 2)"


EXT_KEYS = [
    "extent_margin_h_deg",
    "extent_margin_v_deg",
    "extent_h_deg",
    "extent_v_deg",
    "extent_fade_deg",
    "extent_shape",
]
EXT_DEF = {
    "extent_margin_h_deg": 20.0,
    "extent_margin_v_deg": 25.0,
    "extent_h_deg": None,
    "extent_v_deg": None,
    "extent_fade_deg": 10.0,
    "extent_shape": "ellipse",
}


def load(job_id):
    job = C.get_job(job_id)
    base = [4096, "lanczos", None, None, None]
    ext_vals = [EXT_DEF[k] for k in EXT_KEYS]
    # None 들은 UI 가 숫자 입력을 받을 수 있게 "" 로 (textbox 아니라 Number 지만, Gradio 는 None 허용).
    if job is None:
        return base + ext_vals + ["_Job 없음_"]
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
    ext_vals = [C.get_param(job_id, STAGE, k, EXT_DEF[k]) for k in EXT_KEYS]
    # extent 정보 요약
    md = C.json_md(
        job.path(2, "meta.json"),
        ["extent_h_deg", "extent_v_deg", "extent_fade_deg", "extent_inside_fraction"],
    )
    return [
        int(C.get_param(job_id, STAGE, "size", 4096)),
        C.get_param(job_id, STAGE, "interpolation", "lanczos"),
        slider,
        C.preview_mask(job.path(2, "valid_mask.png"), 1024),
        str(job.path(2, "preview_sbs.png")) if job.path(2, "preview_sbs.png").exists() else None,
    ] + ext_vals + [md]


def build(shared):
    with gr.Tab("2 재투영", id=STAGE):
        with gr.Row():
            size = gr.Dropdown(
                [2048, 3072, 4096, 5120, 6144],
                value=4096,
                label="눈당 해상도 S",
                info="S×S half-equirect 캔버스. Quest 2/3 는 4096~5120 권장. 6144 는 PC 뷰어용.",
                elem_classes=["vr180-tip"],
            )
        # 생성 범위 (extent)
        with gr.Group():
            gr.Markdown("**🎯 생성 범위** — 정면에서 몇 도까지 '제대로' 만들고, 그 바깥을 어둡게 페이드.")
            with gr.Row():
                mh = gr.Slider(
                    0, 50, value=20, step=1, label="수평 여유 margin_h (°, 한쪽)",
                    info="원본 hfov 바깥으로 더 생성할 여유. hfov + 2·margin 이 범위.",
                    elem_classes=["vr180-tip"],
                )
                mv = gr.Slider(
                    0, 50, value=25, step=1, label="수직 여유 margin_v (°, 한쪽)",
                    info="천장·바닥 여유. Quest 시야에 더 들어오므로 조금 넓게.",
                    elem_classes=["vr180-tip"],
                )
            with gr.Row():
                fh = gr.Number(
                    value=None, label="고정 수평 범위 h (°, 선택)",
                    info="주면 margin 대신 사용. 180 이면 전체 생성 (기능 꺼짐).",
                    elem_classes=["vr180-tip"],
                )
                fv = gr.Number(
                    value=None, label="고정 수직 범위 v (°, 선택)",
                    info="주면 margin 대신 사용.",
                    elem_classes=["vr180-tip"],
                )
                fade = gr.Slider(
                    0, 30, value=10, step=0.5, label="페이드 폭 (°)",
                    info="8° 미만이면 깊이 벽이 느껴짐.",
                    elem_classes=["vr180-tip"],
                )
                shape = gr.Dropdown(
                    ["ellipse", "rounded_rect"], value="ellipse", label="모양",
                    info="ellipse 가 비네팅처럼 자연스러움.",
                    elem_classes=["vr180-tip"],
                )
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            interp = gr.Dropdown(
                ["lanczos", "cubic", "linear", "nearest"],
                value="lanczos",
                label="보간",
                info="재투영 샘플링 커널. lanczos = 품질 최고, nearest = 가장 빠름.",
                elem_classes=["vr180-tip"],
            )
        with gr.Row(elem_classes=["vr180-actions"]):
            run_btn = gr.Button(RUN_LABEL, variant="primary", elem_classes=["vr180-run"], scale=2)
            export_btn = gr.Button("📤 Quest 미리보기 내보내기", scale=1)
        extent_info = gr.Markdown()
        slider = gr.ImageSlider(label="L_equi ↔ R_equi — 흰 선 = 생성 범위 경계", type="numpy", interactive=False)
        with gr.Row():
            mask = gr.Image(label="valid_mask", type="numpy", interactive=False)
            prev_file = gr.File(label="preview_sbs.png (다운로드)")
        outputs = [size, interp, slider, mask, prev_file, mh, mv, fh, fv, fade, shape, extent_info]

        def run(job_id, s, it, mh_v, mv_v, fh_v, fv_v, fade_v, shape_v):
            d = {
                "size": int(s),
                "interpolation": it,
                "extent_margin_h_deg": float(mh_v),
                "extent_margin_v_deg": float(mv_v),
                "extent_h_deg": None if fh_v in (None, "") else float(fh_v),
                "extent_v_deg": None if fv_v in (None, "") else float(fv_v),
                "extent_fade_deg": float(fade_v),
                "extent_shape": shape_v,
            }
            yield from C.run_pipeline(job_id, STAGE, STAGE, {STAGE: d})

        def export(job_id):
            job = C.get_job(job_id)
            if job is None or not job.path(2, "preview_sbs.png").exists():
                raise gr.Error("Stage 2 결과가 없습니다")
            dst = job.stage_dir(7) / "preview_stage2_SBS.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(job.path(2, "preview_sbs.png"), dst)
            return f"내보냄: {dst}"

        shared["all_run_buttons"].append(run_btn)

        def _wire():
            others = [b for b in shared["all_run_buttons"] if b is not run_btn]
            run_btn.click(
                C.with_button_lock(run, RUN_LABEL, n_other=len(others)),
                [shared["job_id"], size, interp, mh, mv, fh, fv, fade, shape],
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
        export_btn.click(export, [shared["job_id"]], [shared["log"]])
    return load, outputs
