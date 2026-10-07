"""탭 7 — 출력: 최종 SBS, 파일 경로. 접미사, JPG 품질, XMP. 저장 / 출력 폴더 열기."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 7


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return ["_180_3dh", 95, True, None, None, "_Job 없음_"]
    p = job.stage_params(STAGE)
    suffix = p.get("suffix", "_180_3dh") or ""
    png = job.path(7, f"output_VR180_SBS{suffix}.png")
    jpg = job.path(7, f"output_VR180_SBS{suffix}.jpg")
    files = [str(x) for x in (png, jpg) if x.exists()]
    return [
        suffix,
        int(p.get("jpg_quality", 95)),
        bool(p.get("write_xmp", True)),
        C.preview(png, 2048),
        files or None,
        ("\n".join(f"- `{f}`" for f in files) if files else "_아직 없음_"),
    ]


def build(shared):
    with gr.Tab("7 출력"):
        with gr.Row():
            suffix = gr.Dropdown(
                ["_180_3dh", "_LR", ""], value="_180_3dh", label="파일명 접미사", allow_custom_value=True
            )
            q = gr.Slider(70, 100, value=95, step=1, label="JPG 품질")
            xmp = gr.Checkbox(label="XMP GPano 메타데이터 (JPG)", value=True)
            run_btn = gr.Button("저장 (Stage 7)", variant="primary")
            open_btn = gr.Button("출력 폴더 열기")
        paths_md = gr.Markdown()
        final = gr.Image(label="최종 VR180 SBS", type="numpy", interactive=False)
        files = gr.File(label="다운로드", file_count="multiple")
        outputs = [suffix, q, xmp, final, files, paths_md]

        def run(job_id, s, qq, x):
            yield from C.run_pipeline(
                job_id, STAGE, STAGE, {STAGE: {"suffix": s, "jpg_quality": int(qq), "write_xmp": bool(x)}}
            )

        def open_dir(job_id):
            job = C.get_job(job_id)
            return C.open_folder(job.stage_dir(7) if job else None)

        run_btn.click(run, [shared["job_id"], suffix, q, xmp], [shared["log"], shared["status"]]).then(
            load, [shared["job_id"]], outputs
        )
        open_btn.click(open_dir, [shared["job_id"]], [shared["log"]])
    return load, outputs
