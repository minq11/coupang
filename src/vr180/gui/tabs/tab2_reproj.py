"""탭 2 — 재투영: L/R equirect (원본 밖은 체크무늬), valid mask. 출력 해상도 S, 보간. Quest 미리보기 내보내기."""

from __future__ import annotations

import shutil

import gradio as gr

from vr180.gui import common as C

STAGE = 2


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return [4096, "lanczos", None, None, None]
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
    with gr.Tab("2 재투영"):
        with gr.Row():
            size = gr.Dropdown([2048, 3072, 4096, 5120, 6144], value=4096, label="눈당 해상도 S")
            interp = gr.Dropdown(["lanczos", "cubic", "linear", "nearest"], value="lanczos", label="보간")
            run_btn = gr.Button("실행 (Stage 2 부터)", variant="primary")
            export_btn = gr.Button("Quest 용 미리보기 내보내기 (07_output/preview_stage2_SBS.png)")
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

        run_btn.click(run, [shared["job_id"], size, interp], [shared["log"], shared["status"]]).then(
            load, [shared["job_id"]], outputs
        )
        export_btn.click(export, [shared["job_id"]], [shared["log"]])
    return load, outputs
