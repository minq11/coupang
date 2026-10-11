"""탭 2 — 재투영: L/R equirect (원본 밖은 체크무늬), valid mask. 출력 해상도 S, 보간. Quest 미리보기 내보내기."""

from __future__ import annotations

import shutil

import gradio as gr

from vr180.gui import common as C

STAGE = 2
RUN_LABEL = "▶ 실행 (Stage 2)"


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return [4096, "lanczos", None, None, None]
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
    return [
        int(C.get_param(job_id, STAGE, "size", 4096)),
        C.get_param(job_id, STAGE, "interpolation", "lanczos"),
        slider,
        C.preview_mask(job.path(2, "valid_mask.png"), 1024),
        str(job.path(2, "preview_sbs.png")) if job.path(2, "preview_sbs.png").exists() else None,
    ]


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
        slider = gr.ImageSlider(label="L_equi ↔ R_equi (드래그로 비교)", type="numpy", interactive=False)
        with gr.Row():
            mask = gr.Image(label="valid_mask", type="numpy", interactive=False)
            prev_file = gr.File(label="preview_sbs.png (다운로드)")
        outputs = [size, interp, slider, mask, prev_file]

        def run(job_id, s, it):
            yield from C.run_pipeline(job_id, STAGE, STAGE, {STAGE: {"size": int(s), "interpolation": it}})

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
                [shared["job_id"], size, interp],
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
